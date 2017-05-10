#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl

sess = tf.Session()

# prepare data
X_train, y_train, X_val, y_val, X_test, y_test = \
    tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
# define placeholder
x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x')
y_ = tf.placeholder(tf.int64, shape=[None, ], name='y_')

# define the network
network = tl.layers.InputLayer(x, name='input_layer')
network = tl.layers.Conv2dLayer(network,
                                shape=[3, 3, 1, 32],
                                act=tf.nn.relu,
                                name='conv2d32')
network = tl.layers.MaxPool2d(network, filter_size=(2, 2), name='maxpool1')
network = tl.layers.LocalResponseNormLayer(network, depth_radius=5,
                                           bias=1.0, alpha=0.0001, beta=0.75,
                                           name='lrn1')
network = tl.layers.Conv2dLayer(network,
                                shape=[3, 3, 32, 64],
                                act=tf.nn.relu,
                                name='conv2d64')
network = tl.layers.MaxPool2d(network, filter_size=(2, 2), name='maxpool2')
network = tl.layers.LocalResponseNormLayer(network, depth_radius=5,
                                           bias=1.0, alpha=0.0001, beta=0.75,
                                           name='lrn2')
network = tl.layers.FlattenLayer(network)
network = tl.layers.DenseLayer(network, n_units=128,
                               act=tf.nn.tanh, name='tanh1')
network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
network = tl.layers.DenseLayer(network, n_units=256,
                               act=tf.nn.tanh, name='tanh2')
network = tl.layers.DropoutLayer(network, keep=0.8, name='drop2')

# network = tl.layers.DenseLayer(network, n_units=800,
#                                act=tf.nn.relu, name='relu1')
# network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
# network = tl.layers.DenseLayer(network, n_units=800,
#                                act=tf.nn.relu, name='relu2')
# network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
# the softmax is implemented internally in tl.cost.cross_entropy(y, y_) to
# speed up computation, so we use identity here.
# see tf.nn.sparse_softmax_cross_entropy_with_logits()
network = tl.layers.DenseLayer(network, n_units=10,
                               act=tf.identity,
                               name='output_layer')

# define cost function and metric.
y = network.outputs
cost = tl.cost.cross_entropy(y, y_, name='xentropy')

correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(tf.nn.softmax(y), 1)

# define the optimizer
train_params = network.all_params
train_op = tf.train.AdamOptimizer(
    learning_rate=0.001,  # beta1=0.9, beta2=0.999,
    epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

# initialize all variables in the session
tl.layers.initialize_global_variables(sess)

# print network information
# network.print_params()
network.print_layers()

# train the network
tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
             acc=acc, batch_size=100, n_epoch=20, print_freq=1,
             X_val=X_val, y_val=y_val, eval_train=False)

# evaluation
tl.utils.test(sess, network, acc, X_test, y_test,
              x, y_, batch_size=None, cost=cost)

# save the network to .npz file
# tl.files.save_npz(network.all_params, name='model.npz')
sess.close()
