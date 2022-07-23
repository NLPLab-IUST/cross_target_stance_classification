#!/bin/bash

declare -a topics=("dt" "hc" "fm" "la" "a" "cc")
for topic in  "${topics[@]}";
do
     echo $topic
     python3 train_model_CrossNet_keras.py -tr_te --target cc_cc --n_aspect 1 --bsize 128 --rnn_dim 128 --dense_dim 64 --dropout_rate 0.2 --max_epoch 200 --learning_rate 0.001 ;
done