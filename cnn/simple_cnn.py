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
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, global_max_pool, conv_2d, max_pool_1d
from tflearn.layers.merge_ops import merge
from tflearn.layers.estimator import regression
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
network = input_data(shape=[None, 100], name='input')
network = tflearn.embedding(network, input_dim=10000, output_dim=128)

# branch1 = conv_1d(network, 5, 2, padding='valid',
#                   activation='relu', regularizer="L2")
# branch2 = conv_1d(network, 5, 3, padding='valid',
#                   activation='relu', regularizer="L2")
# branch3 = conv_1d(network, 5, 4, padding='valid',
#                   activation='relu', regularizer="L2")

network = tf.expand_dims(network, -1)
conv1 = conv_2d(network, nb_filter=2, filter_size=2,
                padding='valid', activation='relu')
conv2 = conv_2d(network, nb_filter=2, filter_size=3,
                padding='valid', activation='relu')
conv3 = conv_2d(network, nb_filter=2, filter_size=4,
                padding='valid', activation='relu')
pool1 = global_max_pool(conv1)
pool2 = global_max_pool(conv2)
pool3 = global_max_pool(conv3)

network = merge([pool1, pool2, pool3], mode='concat', axis=1)
# import pdb
# pdb.set_trace()

# network = tf.expand_dims(network, 2)
# network = global_max_pool(network)
network = dropout(network, 0.5)
network = fully_connected(network, 2, activation='softmax')
network = regression(network, optimizer='adam', learning_rate=0.001,
                     loss='categorical_crossentropy', name='target')
# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit(trainX, trainY, n_epoch=5, shuffle=True, validation_set=(
    testX, testY), show_metric=True, batch_size=32)
