**This is a keras implementation of CrossNet for paper Cross-Target Stance Classification with Self-Attention Networks (https://arxiv.org/abs/1805.06593)**

## The experimental process consists of two steps:

1.   preprocessing (data_util.py and tokenizer): convert tweet text, target phrase, and label into internal matrix format, with shapes (batch_size, sent_length), (batch_size, target_length), (batch_size, num_class)
2.   training and testing (train_model_CrossNet_keras.py)
  config.py: all directory and model configurations are here
  models/CrossNet.py: model implementation of CrossNet
  models/layers.py: layer implementation of CrossNet


## getting started:

First of all you need to download **glove.twitter.27B.200d** from kaggle (https://www.kaggle.com/datasets/larryfreeman/glove-twitter-27b-200d-txt)

### Requirements:
  * python 3.7
  * keras 2.1.3
  * tensorflow 1.13.1
  * gensim 2.5
  * nltk 3.7
  * pandas 1.1.5
  * numpy 1.16.1
  * sklearn 0.0
  * h5py 2.10.0

you can use the following command to install all dependencies:


```
pip install -r requirements.txt
```


## Usage:

First of all you should change a file. go to your_venv_name/lib/python3.7/site-packages/gensim/models/ldamodel.py and line 56:
change **from scipy.misc import logsumexp** to **from scipy.special import logsumexp**

In the second step, create cache and model folder in the folder named data. Then, run data_util.py to fill those folders.

On windows (Train and test):


1.   set PYTHONPATH=%PYTHONPATH%;C:\path_to_project\cross_target_stance_classification\
2.   C:\path_to_python\python.exe C:\path_to_project\cross_target_stance_classification\train_model_CrossNet_keras.py -tr_te --target cc_cc --n_aspect 1 --bsize 128 --rnn_dim 128 --dense_dim 64 --dropout_rate 0.2 --max_epoch 200 --learning_rate 0.001

**OR** you can use the following code command to train and test all targets:

'''
bash ./runModel.bash
'''

**NOTE:** you can change -tr_te to train, test and ts(test_single_stance) in order to use other functionalities.
