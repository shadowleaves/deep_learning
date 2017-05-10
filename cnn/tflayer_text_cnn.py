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
trainX, trainY = train
testX, testY = test

# Data preprocessing
# Sequence padding
trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(trainY, nb_classes=2)
testY = to_categorical(testY, nb_classes=2)

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

network = tl.layers.ConcatLayer([branch1, branch2, branch3],
                                concat_dim=1, name='concat_layer')

network = tl.layers.ExpandDimsLayer(network, axis=2, name='expand_dims')
k = network.all_layers[-1].shape[1].value
network = tl.layers.MaxPool2d(network,
                              filter_size=[k, 1],
                              # filter_size=1,
                              # strides=1,
                              # padding='valid',
                              )
network = tl.layers.ReshapeLayer(network, shape=[-1, nbf])
network = tl.layers.DropoutLayer(network, keep=0.5)
network = tl.layers.DenseLayer(network, n_units=2, act=tf.nn.softmax)
network.print_layers()

# define cost function and metric.
y = network.outputs
# y_ = tf.reshape(y_, [32, 2])
# y = tf.reshape(y, [32, 2])
import pdb
pdb.set_trace()
# y_op = tf.argmax(tf.nn.softmax(y), 1)
cost = tl.cost.cross_entropy(y, y_, 'cost')
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# define the optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(
    learning_rate=0.001,  # beta1=0.9, beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
).minimize(cost, var_list=train_params)


# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

# print network information
network.print_params()
network.print_layers()

# train the network
tl.utils.fit(sess, network, train_op, cost, trainX, trainY, x, y_,
             # acc=acc,
             batch_size=32, n_epoch=5, print_freq=1,
             X_val=testX, y_val=testY,
             eval_train=False)

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
# model.fit(trainX, trainY, n_epoch=5, shuffle=True, validation_set=(
#     testX, testY), show_metric=True, batch_size=32)
