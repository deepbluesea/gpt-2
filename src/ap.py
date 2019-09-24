# Original work Copyright 2018 The Google AI Language Team Authors.
# Modified work Copyright 2019 Rowan Zellers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import re
import tensorflow as tf

# Original work Copyright 2018 The Google AI Language Team Authors.
# Modified work Copyright 2019 Rowan Zellers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import re

import six
import tensorflow as tf
import numpy as np
from tensorflow.python.lib.io import file_io


def _save_np(absolute_fn, array):
    if absolute_fn.startswith('gs://'):
        with file_io.FileIO(absolute_fn, 'w') as f:
            np.save(f, array)
    else:
        np.save(absolute_fn, array)


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.
    Args:
      tensor: A tf.Tensor to check the rank of.
      expected_rank: Python integer or list of integers, expected rank.
      name: Optional name of the tensor for the error message.
    Raises:
      ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        name = tensor.name

    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.get_variable_scope().name
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.
    Args:
      tensor: A tf.Tensor object to find the shape of.
      expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
      name: Optional name of the tensor for the error message.
    Returns:
      A list of dimensions of the shape of tensor. All static dimensions will
      be returned as python integers, and dynamic dimensions will be returned
      as tf.Tensor scalars.
    """
    if name is None:
        name = tensor.name

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def gelu(input_tensor):
    """Gaussian Error Linear Unit.
    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415
    Args:
      input_tensor: float Tensor to perform activation.
    Returns:
      `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor * cdf


def layer_norm(input_tensor, name=None, epsilon=1e-5):
    """Run layer normalization on the last dimension of the tensor."""
    name2use = f'LayerNorm_{name}' if name is not None else name
    with tf.variable_scope(name2use, default_name='LayerNorm'):
        dim = input_tensor.shape[-1].value
        gamma = tf.get_variable('gamma', [dim], initializer=tf.constant_initializer(1))
        beta = tf.get_variable('beta', [dim], initializer=tf.constant_initializer(0))
        mean = tf.reduce_mean(input_tensor, axis=-1, keepdims=True)
        std = tf.reduce_mean(tf.square(input_tensor - mean), axis=-1, keepdims=True)
        input_tensor = (input_tensor - mean) * tf.rsqrt(std + epsilon)
        input_tensor = input_tensor * gamma + beta
    return input_tensor


def dropout(input_tensor, dropout_prob):
    """Perform dropout.
    Args:
      input_tensor: float Tensor.
      dropout_prob: Python float. The probability of dropping out a value (NOT of
        *keeping* a dimension as in `tf.nn.dropout`).
    Returns:
      A version of `input_tensor` with dropout applied.
    """
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor
    output = tf.nn.dropout(input_tensor, rate=dropout_prob)
    return output


def get_attention_mask(nd, ns, *, dtype):
    """
    this is a TPU compatible version of tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd)
    where the lower right triangle contains 1s
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1
    return (assignment_map, initialized_variable_names)


def construct_scalar_host_call(metric_dict, model_dir, prefix=""):
    """Construct a host call to log scalars when training on TPU.
    Args:
      metric_dict: A dict of the tensors to be logged.
      model_dir: The location to write the summary.
      prefix: The prefix (if any) to prepend to the metric names.
    Returns:
      A tuple of (function, args_to_be_passed_to_said_function)
    """
    metric_names = list(metric_dict.keys())

    def host_call_fn(global_step, *args):
        """Training host call. Creates scalar summaries for training metrics.
        This function is executed on the CPU and should not directly reference
        any Tensors in the rest of the `model_fn`. To pass Tensors from the
        model to the `metric_fn`, provide as part of the `host_call`. See
        https://www.tensorflow.org/api_docs/python/tf/contrib/tpu/TPUEstimatorSpec
        for more information.
        Arguments should match the list of `Tensor` objects passed as the second
        element in the tuple passed to `host_call`.
        Args:
          global_step: `Tensor with shape `[batch]` for the global_step
          *args: Remaining tensors to log.
        Returns:
          List of summary ops to run on the CPU host.
        """
        step = global_step[0]
        with tf.contrib.summary.create_file_writer(
                logdir=model_dir, filename_suffix=".host_call").as_default():
            with tf.contrib.summary.always_record_summaries():
                for i, name in enumerate(metric_names):
                    tf.contrib.summary.scalar(prefix + name, args[i][0], step=step)

                return tf.contrib.summary.all_summary_ops()

    # To log the current learning rate, and gradient norm for Tensorboard, the
    # summary op needs to be run on the host CPU via host_call. host_call
    # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
    # dimension. These Tensors are implicitly concatenated to
    # [params['batch_size']].
    global_step_tensor = tf.reshape(
        tf.compat.v1.train.get_or_create_global_step(), [1])
    other_tensors = [tf.reshape(metric_dict[key], [1]) for key in metric_names]

    return host_call_fn, [global_step_tensor] + other_tensors

def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps):
    """Creates an optimizer training op."""
    global_step = tf.train.get_or_create_global_step()

    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

    # Implements linear decay of the learning rate.
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
        global_steps_int = tf.cast(global_step, tf.int32)
        warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

        global_steps_float = tf.cast(global_steps_int, tf.float32)
        warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

        warmup_percent_done = global_steps_float / warmup_steps_float
        warmup_learning_rate = init_lr * warmup_percent_done

        is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
        learning_rate = (
                (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

    # It is recommended that you use this optimizer for fine tuning, since this
    # is how the model was trained (note that the Adam m/v variables are NOT
    # loaded from init_checkpoint.)
    optimizer = AdaFactorOptimizer(
        learning_rate=learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])



    tvars = tf.trainable_variables()
    grads = tf.gradients(loss, tvars)

    # You could do this, but instead we don't because a) it's slow and b) we already did the 'update clipping'
    # (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

    train_op = optimizer.apply_gradients(
        zip(grads, tvars), global_step=global_step)

    # Normally the global step update is done inside of `apply_gradients`.
    # However, `AdaFactorOptimizer` doesn't do this. But if you use
    # a different optimizer, you should probably take this line out.
    new_global_step = global_step + 1
    train_op = tf.group(train_op, [global_step.assign(new_global_step)])

    train_metrics = {
        'learning_rate': learning_rate,
        'minibatch_loss': loss,
        # 'minibatch_ppl': tf.math.exp(loss),
    }
    return train_op, train_metrics


class AdaFactorOptimizer(tf.train.Optimizer):
    """here's the optimizer we'll use"""

    def __init__(self,
                 learning_rate,
                 weight_decay_rate=0.0,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-6,
                 exclude_from_weight_decay=None,
                 clipping_rate=1.0,
                 name="AdaFactorOptimizer"):
        """Constructs a AdaFactorOptimizer."""
        super(AdaFactorOptimizer, self).__init__(False, name)

        self.learning_rate = learning_rate
        self.weight_decay_rate = weight_decay_rate
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.epsilon1 = 1e-30
        self.epsilon2 = 0.001
        self.clipping_rate = clipping_rate
        self.exclude_from_weight_decay = exclude_from_weight_decay
        self.use_locking = False

    def _use_factored(self, shape):
        return len(shape) >= 2

    def _parameter_scale(self, var):
        """Estimate the scale of the parameters from the current values.
        We include a minimum value of 0.001 to give it a chance to escape 0
        if it was zero-initialized.
        Instead of using the value, we could impute the scale from the shape,
        as initializers do.
        Args:
          var: a variable or Tensor.
        Returns:
          a Scalar
        """
        return tf.maximum(reduce_rms(var), self.epsilon2)

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        """See base class."""
        assignments = []
        for (grad, param) in grads_and_vars:
            if grad is None or param is None:
                continue

            param_name = self._get_variable_name(param.name)
            shape_list = get_shape_list(param, expected_rank=[1, 2])

            # decay_rate = 1 - tf.pow(tf.cast(tf.train.get_or_create_global_step(), tf.float32) + 1.0, -0.8)
            decay_rate = self.beta_2
            grad_squared = tf.square(grad) + self.epsilon1

            update_scale = self.learning_rate
            # update_scale = self.learning_rate * tf.cast(self._parameter_scale(param), dtype=tf.float32)

            # HACK: Make things dependent on grad.
            # This confounds the XLA rewriter and keeps it from fusing computations
            # across different variables.  This fusion is a bad for HBM usage, since
            # it causes the gradients to persist in memory.
            grad_squared_mean = tf.reduce_mean(grad_squared)
            decay_rate += grad_squared_mean * 1e-30
            update_scale += grad_squared_mean * 1e-30

            # END HACK

            if self._use_factored(shape_list):
                num_rows, num_columns = shape_list

                vr = tf.get_variable(
                    name=param_name + "/adafactor_vr",
                    shape=[num_rows],
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer())
                vc = tf.get_variable(
                    name=param_name + "/adafactor_vc",
                    shape=[num_columns],
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer())

                next_vr = decay_rate * vr + (1 - decay_rate) * tf.reduce_mean(grad_squared, 1)
                next_vc = decay_rate * vc + (1 - decay_rate) * tf.reduce_mean(grad_squared, 0)

                long_term_mean = tf.reduce_mean(next_vr, -1, keepdims=True)
                r_factor = tf.rsqrt(next_vr / long_term_mean + self.epsilon1)
                c_factor = tf.rsqrt(next_vc + self.epsilon1)
                update = grad * tf.expand_dims(r_factor, -1) * tf.expand_dims(c_factor, -2)

                assignments.append(vr.assign(next_vr, use_locking=self.use_locking))
                assignments.append(vc.assign(next_vc, use_locking=self.use_locking))
            else:
                v = tf.get_variable(
                    name=param_name + "/adafactor_v",
                    shape=shape_list,
                    dtype=tf.float32,
                    trainable=False,
                    initializer=tf.zeros_initializer())
                next_v = decay_rate * v + (1 - decay_rate) * grad_squared

                assignments.append(v.assign(next_v, use_locking=self.use_locking))
                update = grad * tf.rsqrt(next_v + self.epsilon1)

            clipping_denom = tf.maximum(1.0, reduce_rms(update) / self.clipping_rate)
            update /= clipping_denom

            # Do weight decay
            # Just adding the square of the weights to the loss function is *not*
            # the correct way of using L2 regularization/weight decay with Adam,
            # since that will interact with the m and v parameters in strange ways.
            #
            # Instead we want ot decay the weights in a manner that doesn't interact
            # with the m/v parameters. This is equivalent to adding the square
            # # of the weights to the loss with plain (non-momentum) SGD.
            if self._do_use_weight_decay(param_name):
                update += self.weight_decay_rate * param

            update_with_lr = update_scale * update
            next_param = param - update_with_lr

            assignments.append(param.assign(next_param, use_locking=self.use_locking))
        return tf.group(*assignments, name=name)

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if not self.weight_decay_rate:
            return False
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name


def reduce_rms(x):
    return tf.sqrt(tf.reduce_mean(tf.square(x)))
