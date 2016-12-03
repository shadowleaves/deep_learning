#!/usr/bin/env python
import minpy.numpy as np
# from minpy.core import grad
# import tensorflow as tf

sigmoid = lambda x: 1. / (1 + np.exp(-x))


def rnn_step(x, h, wx, wh, b):
    return sigmoid(np.dot(x, wx) + np.dot(h, wh) + b)


def loss(pred, y):
    return 0.5 * (pred - y)**2

if __name__ == '__main__':

    X = np.array([18., 9., -8.]).reshape(3, 1)
    h = np.array([0])
    wx = np.array([-0.1])
    wh = np.array([0.5])
    b = np.array([0.4])
    wa = np.array([0.25])
    ba = np.array([0])
    T = np.array([0.1, -0.1, 0.2]).reshape(3, 1)

    # X = np.array([9., 4., -2.]).reshape(3, 1)
    # h = np.array([0])
    # wx = np.array([0.5])
    # wh = np.array([-1.])
    # b = np.array([-1.])
    # wa = np.array([-0.7])
    # ba = np.array([0])

    states = []
    output = []
    for i in xrange(3):
        h = rnn_step(X[i], h, wx, wh, b)
        o = np.dot(h, wa) + ba
        states.append(h)
        output.append(o)

    # states = np.asarray(states).reshape(3, 1)
    # output = np.asarray(output).reshape(3, 1)
    print 'hidden state: ', states
    print 'output      : ', output

    y = output
    dev =  -(T[2]-y[2]) * wa * h * (1 - h)
    print dev
    # print np.round(dev, 4)
    # print np.round(states)
    pass
