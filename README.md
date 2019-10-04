import os
os.environ['TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'] = '1'

!PYTHONPATH=src python ./train.py --dataset aqxx.txt --model_name 774M --batch_size 1 --learning_rate 0.0003 --memory_saving_gradients --only_train_transformer_layers --optimizer b --save_every 10000 --sample_every 50 --sample_length 1023 --train_vars_limit --train_vars 408
