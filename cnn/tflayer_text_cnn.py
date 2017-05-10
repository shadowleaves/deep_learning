#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simple example using convolutional neural network to classify IMDB
sentiment dataset.
References:
    - Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng,
    and Christopher Potts. (2011). Learning Word Vectors for Sentiment
    Analysis. The 49th Annual Meeting of the Association for Computational
    Linguistics (ACL 2011).
    - Kim Y. Convolutional Neural Networks for Sentence Classification[C].
    Empirical Methods in Natural Language Processing, 2014.
Links:
    - http://ai.stanford.edu/~amaas/data/sentiment/
    - http://emnlp2014.org/papers/pdf/EMNLP2014181.pdf
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import tensorlayer as tl
import numpy as np

# from tensorlayer.prepro import pad_sequences
# import tflearn
# from tflearn.layers.core import input_data, dropout, fully_connected
# from tflearn.layers.conv import conv_1d, global_max_pool
# from tflearn.layers.merge_ops import merge
# from tflearn.layers.estimator import regression
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

# IMDB Dataset loading
train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000,
                                valid_portion=0.1)
X_train, y_train = train
X_val, y_val = test

# Data preprocessing
# Sequence padding
X_train = pad_sequences(X_train, maxlen=100, value=0.)
X_val = pad_sequences(X_val, maxlen=100, value=0.)

y_train = np.array(y_train, dtype='int32')
y_val = np.array(y_val, dtype='int32')
# Converting labels to binary vectors
# y_train = to_categorical(y_train, nb_classes=2)
# Y_val = to_categorical(Y_val, nb_classes=2)

# Building convolutional network
# embedding
sess = tf.InteractiveSession()

embd_dims = 128
nbf = 128  # doesn't have to be equal to embedding dims
x = tf.placeholder(tf.int32, shape=[None, 100], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')
# network = tl.layers.InputLayer(x, name='input')
network = tl.layers.EmbeddingInputlayer(inputs=x,
                                        vocabulary_size=10000,
                                        embedding_size=embd_dims,
                                        name='embedding_layer')

branch1 = tl.layers.Conv1dLayer(network,
                                act=tf.nn.relu,
                                shape=[3, nbf, nbf],
                                stride=1,
                                padding='VALID',
                                name='branch1',
                                )
branch2 = tl.layers.Conv1dLayer(network,
                                act=tf.nn.relu,
                                shape=[4, nbf, nbf],
                                stride=1,
                                padding='VALID',
                                name='branch2',
                                )
branch3 = tl.layers.Conv1dLayer(network,
                                act=tf.nn.relu,
                                shape=[5, nbf, nbf],
                                stride=1,
                                padding='VALID',
                                name='branch3',
                                )
# reg1 = tf.contrib.layers.l2_regularizer(0.01)(branch1.all_layers[-1])
# reg2 = tf.contrib.layers.l2_regularizer(0.01)(branch2.all_layers[-1])
# reg3 = tf.contrib.layers.l2_regularizer(0.01)(branch3.all_layers[-1])

network = tl.layers.ConcatLayer([branch1, branch2, branch3],
                                concat_dim=1, name='concat_layer')

network = tl.layers.ExpandDimsLayer(network, axis=3, name='expand_dims')
shape = [z.value if z.value else -1 for z in
         network.all_layers[-1].shape.dims[:-1]]
network = tl.layers.ReshapeLayer(network, shape=shape)
# network = tl.layers.ExpandDimsLayer(network, axis=3, name='expand_dims')
k = network.all_layers[-1].shape[1].value

network = tl.layers.MaxPool1d(network,
                              # filter_size=[k, 1],
                              filter_size=k,
                              strides=1,
                              # padding='valid',
                              )
network = tl.layers.FlattenLayer(network)
network = tl.layers.DropoutLayer(network, keep=0.5)
network = tl.layers.DenseLayer(network, n_units=2, act=tf.identity)
network.print_layers()

# define cost function and metric.
y = network.outputs
# y_ = tf.reshape(y_, [32, 2])
# y = tf.reshape(y, [32, 2])
# y_op = tf.argmax(tf.nn.softmax(y), 1)

cost = tl.cost.cross_entropy(y, y_, 'cost')  # + reg1 + reg2 + reg3
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)

# define the optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(
    learning_rate=0.001, beta1=0.9, beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
).minimize(cost, var_list=train_params)


# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

# print network information
network.print_params()
network.print_layers()

# train the network
tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
             acc=acc,
             batch_size=32, n_epoch=5, print_freq=1,
             X_val=X_val, y_val=y_val,
             eval_train=False)

sess.close()
# import pdb
# pdb.set_trace()
# y = network.outpu
# y = network.outputs
# cost = tl.cost.cross_entropy(y, y_, 'cost')
# import pdb
# pdb.set_trace()
# # import pdb
# # pdb.set_trace()
# network = regression(network, optimizer='adam', learning_rate=0.001,
#                      loss='categorical_crossentropy', name='target')
# # Training
# model = tflearn.DNN(network, tensorboard_verbose=0)
# model.fit(X_train, y_train, n_epoch=5, shuffle=True, validation_set=(
#     X_val, Y_val), show_metric=True, batch_size=32)
