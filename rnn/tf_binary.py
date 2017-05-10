#!/usr/bin/env python
'''
A Dynamic Recurrent Neural Network (LSTM) implementation example using
TensorFlow library. This example is using a toy dataset to classify linear
sequences. The generated sequences have variable length.
Long Short Term Memory paper:
http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
# import random
import numpy as np


# Show an example input and target
def printSample(x1, x2, t, y=None):
    """Print a sample in a more visual way."""
    x1 = ''.join([str(int(d)) for d in x1])
    x2 = ''.join([str(int(d)) for d in x2])
    t = ''.join([str(int(d[0])) for d in t])
    if y is not None:
        y = ''.join([str(int(d[0])) for d in y])
    print('x1:   {:s}   {:2d}'.format(x1, int(''.join(reversed(x1)), 2)))
    print('x2: + {:s}   {:2d} '.format(x2, int(''.join(reversed(x2)), 2)))
    print('      -------   --')
    print('t:  = {:s}   {:2d}'.format(t, int(''.join(reversed(t)), 2)))
    if y is not None:
        print('y:  = {:s}   {:2d}'.format(y, int(''.join(reversed(y)), 2)))
    print('\n')

# ====================
#  TOY DATA GENERATOR
# ====================


class ToySequenceData(object):
    """ Generate sequence of data with dynamic length.
    This class generate samples for training:
    - Class 0: linear sequences (i.e. [0, 1, 2, 3,...])
    - Class 1: random sequences (i.e. [1, 3, 10, 7,...])
    NOTICE:
    We have to pad each sequence to reach 'max_seq_len' for TensorFlow
    consistency (we cannot feed a numpy array with inconsistent
    dimensions). The dynamic calculation will then be perform thanks to
    'seqlen' attribute that records every actual sequence length.
    """

    def __init__(self, n_samples=1000, seq_len=7):
        """Create a dataset for binary addition and return as input, targets.
        """
        max_int = 2**(seq_len - 1)  # Maximum integer that can be added
        # Transform integer in binary format
        format_str = '{:0' + str(seq_len) + 'b}'
        X = np.zeros((n_samples, seq_len, 2))  # Input samples
        T = np.zeros((n_samples, seq_len, 1))  # Target samples
        self.batch_id = 0

        # Fill up the input and target matrix
        for i in xrange(n_samples):
            # Generate random numbers to add
            nb1 = np.random.randint(0, max_int)
            nb2 = np.random.randint(0, max_int)
            # Fill current input and target row.
            # Note that binary numbers are added from right to left,
            # but our RNN reads from left to right, so reverse the sequence.
            X[i, :, 0] = list(
                reversed([int(b) for b in format_str.format(nb1)]))
            X[i, :, 1] = list(
                reversed([int(b) for b in format_str.format(nb2)]))
            T[i, :, 0] = list(
                reversed([int(b) for b in format_str.format(nb1 + nb2)]))
        # return X, T
        self.data = X
        self.labels = T
        # self.data = []
        # self.labels = []
        # self.seqlen = []

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0

        next_id = min(self.batch_id + batch_size, len(self.data))
        batch_data = (self.data[self.batch_id:next_id])
        batch_labels = (self.labels[self.batch_id:next_id])
        self.batch_id = next_id

        return batch_data, batch_labels


def rnn_step(x, state, wx, wh, b):
    """RNN loop"""
    return tf.tanh(tf.matmul(x, wx) + tf.matmul(state, wh) + b)


def RNN(x, wx, wh, b, wa, ba,
        n_steps, n_inputs, n_hidden, n_outputs):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    batch_size = x.get_shape()[0].value
    state = tf.Variable(tf.zeros([batch_size, n_hidden]))
    # output = tf.Variable(tf.zeros([n_hidden, n_outputs]))
    output = []

    # raise NotImplementedError
    # Define a lstm cell with tensorflow
    for i in xrange(n_steps):
        x_step = x[:, i, :]
        # state = tf.tanh(tf.matmul(x_step, wx) + tf.matmul(state, wh) + b)
        state = rnn_step(x_step, state, wx, wh, b)
        z = tf.matmul(state, wa) + ba
        # z = tf.sigmoid(z)
        output.append(z)

    output = tf.transpose(tf.stack(output), [1, 0, 2])
    return output


def xavier(shape, coef=1.):
    assert len(shape) == 2
    n_in, n_out = shape
    a = np.sqrt(6.0 / (n_in + n_out)) * coef
    res = tf.random_uniform(shape, minval=-a, maxval=a, dtype=tf.float32)
    return tf.Variable(res)


def main():

    # ==========
    #   MODEL
    # ==========

    # Parameters
    # learning_rate = 0.05
    # training_iters = 1000000
    batch_size = 100
    display_step = 100

    # Network Parameters
    # seq_max_len = 20  # Sequence max length
    n_inputs = 2
    n_hidden = 3  # hidden layer num of features
    n_outputs = 1  # linear sequence or not

    # Create training samples
    seed = 5
    np.random.seed(seed=seed)
    tf.set_random_seed(seed)
    train_size = 2000
    test_size = 100
    seq_len = 7
    trainset = ToySequenceData(n_samples=train_size, seq_len=seq_len)
    testset = ToySequenceData(n_samples=test_size, seq_len=seq_len)

    # tf Graph input
    x = tf.placeholder("float", [batch_size, seq_len, n_inputs])
    y = tf.placeholder("float", [batch_size, seq_len, n_outputs])

    # Define weights
    wx = xavier((n_inputs, n_hidden))
    # bx = tf.Variable(tf.zeros([n_hidden, ]))

    wh = xavier((n_hidden, n_hidden))
    b = tf.Variable(tf.zeros([n_hidden, ]))

    wa = xavier((n_hidden, n_outputs))
    ba = tf.Variable(tf.zeros([n_outputs, ]))

    # momentum = tf.Variable(tf.random_normal([n_hidden, n_outputs]))

    pred = RNN(x, wx, wh, b, wa, ba,
               n_steps=seq_len,
               n_inputs=n_inputs,
               n_hidden=n_hidden,
               n_outputs=n_outputs,)

    # Define loss and optimizer
    # cost = cost_func(pred, y)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        logits=pred, labels=y))

    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=0.02,
        decay=0.5,
        momentum=0.8,
        epsilon=1e-6,
        # centered=True,
    ).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.round(tf.sigmoid(pred)), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    # epoch = 0
    n_epochs = 100
    from utils.timedate import timing

    with tf.Session() as sess:
        sess.run(init)

        # step = 1
        t0 = timing()
        for epoch in xrange(n_epochs):
            print ('epoch: %d' % epoch)
            # Keep training until reach max iterations
            # while step * batch_size < training_iters:
            for step in xrange(train_size / batch_size):
                batch_x, batch_y = trainset.next(batch_size)
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

                if step % display_step == 0:
                    # Calculate batch accuracy
                    yhat = sess.run(pred, feed_dict={x: batch_x})

                    acc = sess.run(accuracy, feed_dict={
                                   x: batch_x, y: batch_y})
                    # Calculate batch loss
                    loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                    print(
                        "Iter " + str(step * batch_size) + ", Loss= " +
                        "{:.6f}".format(loss) + ", Training Acc= " +
                        "{:.5f}".format(acc))
                step += 1

        print("Optimization Finished!")
        timing(t0, 'tensorflow')

        # Calculate accuracy
        test_data = testset.data
        test_label = testset.labels
        # test_seqlen = testset.seqlen
        print("Testing Accuracy:",
              sess.run(accuracy, feed_dict={x: test_data, y: test_label}))


if __name__ == '__main__':
    main()
