""" Vanilla RNN
Parallelizes scan over sequences by using mini-batches.
@author Graham Taylor
"""
import numpy as np
import theano
# import theano.tensor as T
# from sklearn.base import BaseEstimator
import logging
import time
# import os
# import datetime
# import cPickle as pickle
from rnn_minibatch import MetaRNN

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
plt.ion()

mode = theano.Mode(linker='cvm')
#mode = 'DEBUG_MODE'


def create_dataset(nb_samples, sequence_len):
    """Create a dataset for binary addition and return as input, targets."""
    max_int = 2**(sequence_len - 1)  # Maximum integer that can be added
    # Transform integer in binary format
    format_str = '{:0' + str(sequence_len) + 'b}'
    nb_inputs = 2  # Add 2 binary numbers
    nb_outputs = 1  # Result is 1 binary number
    X = np.zeros((nb_samples, sequence_len, nb_inputs))  # Input samples
    T = np.zeros((nb_samples, sequence_len, nb_outputs))  # Target samples
    # Fill up the input and target matrix
    for i in xrange(nb_samples):
        # Generate random numbers to add
        nb1 = np.random.randint(0, max_int)
        nb2 = np.random.randint(0, max_int)
        # Fill current input and target row.
        # Note that binary numbers are added from right to left,
        # but our RNN reads from left to right, so reverse the sequence.
        X[i, :, 0] = list(reversed([int(b) for b in format_str.format(nb1)]))
        X[i, :, 1] = list(reversed([int(b) for b in format_str.format(nb2)]))
        T[i, :, 0] = list(reversed([int(b)
                                    for b in format_str.format(nb1 + nb2)]))
    return X, T


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
    print '\n'


# def test_real(n_epochs=1000):
#     """ Test RNN with real-valued outputs. """
#     n_hidden = 10
#     n_in = 5
#     n_out = 3
#     n_steps = 10
#     n_seq = 10  # per batch
#     n_batches = 10

#     np.random.seed(0)
#     # simple lag test
#     seq = np.random.randn(n_steps, n_seq * n_batches, n_in)
#     targets = np.zeros((n_steps, n_seq * n_batches, n_out))

#     targets[1:, :, 0] = seq[:-1, :, 3]  # delayed 1
#     targets[1:, :, 1] = seq[:-1, :, 2]  # delayed 1
#     targets[2:, :, 2] = seq[:-2, :, 0]  # delayed 2

#     targets += 0.01 * np.random.standard_normal(targets.shape)

#     model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
#                     learning_rate=0.01, learning_rate_decay=0.999,
#                     n_epochs=n_epochs, batch_size=n_seq, activation='tanh',
#                     L2_reg=1e-3)

#     model.fit(seq, targets, validate_every=100, optimizer='bfgs')

#     plt.close('all')
#     fig = plt.figure()
#     ax1 = plt.subplot(211)
#     plt.plot(seq[:, 0, :])
#     ax1.set_title('input')
#     ax2 = plt.subplot(212)
#     true_targets = plt.plot(targets[:, 0, :])

#     guess = model.predict(seq[:, 0, :][:, np.newaxis, :])

#     guessed_targets = plt.plot(guess.squeeze(), linestyle='--')
#     for i, x in enumerate(guessed_targets):
#         x.set_color(true_targets[i].get_color())
#     ax2.set_title('solid: true output, dashed: model output')


def test_binary(multiple_out=False, n_epochs=100, optimizer='sgd'):
    """ Test RNN with binary outputs. """
    # n_hidden = 10
    # n_in = 5
    # if multiple_out:
    #     n_out = 2
    # else:
    #     n_out = 1
    # n_steps = 10
    # n_seq = 10  # per batch
    # n_batches = 50

    # np.random.seed(0)
    # # simple lag test
    # seq = np.random.randn(n_steps, n_seq * n_batches, n_in)
    # targets = np.zeros((n_steps, n_seq * n_batches, n_out))

    # # whether lag 1 (dim 3) is greater than lag 2 (dim 0)
    # targets[2:, :, 0] = np.cast[np.int](seq[1:-1, :, 3] > seq[:-2, :, 0])

    # Create dataset
    nb_train = 2000  # Number of training samples
    nb_test = 500
    # Addition of 2 n-bit numbers can result in a n+1 bit number
    sequence_len = 7  # Length of the binary sequence

    np.random.seed(seed=5)
    # from numpy_binary import create_dataset  # , printSample
    X_train, T_train = create_dataset(nb_train, sequence_len)
    X_test, T_test = create_dataset(nb_test, sequence_len)

    model = MetaRNN(n_in=2, n_hidden=3, n_out=1,
                    learning_rate=0.05, learning_rate_decay=1,
                    initial_momentum=0.8,
                    momentum_switchover=1e6,
                    n_epochs=n_epochs, activation='tanh', output_type='binary',
                    # interactive=True
                    # mode=theano.Mode(linker='cvm')
                    )

    model.fit(X_train, T_train, validate_every=1, compute_zero_one=True,
              optimizer=optimizer)

    # model.fit(X_train, T_train, validation_frequency=100)

    # if multiple_out:
    #     # whether product of lag 1 (dim 4) and lag 1 (dim 2)
    #     # is less than lag 2 (dim 0)
    #     targets[2:, :, 1] = np.cast[np.int](
    #         (seq[1:-1, :, 4] * seq[1:-1, :, 2]) > seq[:-2, :, 0])

    # model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
    #                 learning_rate=0.005, learning_rate_decay=0.999,
    #                 n_epochs=n_epochs, batch_size=n_seq, activation='tanh',
    #                 output_type='binary')

    # model.fit(seq, targets, validate_every=100, compute_zero_one=True,
    #           optimizer=optimizer)

    # seqs = xrange(10)

    # plt.close('all')
    # for seq_num in seqs:
    #     fig = plt.figure()
    #     ax1 = plt.subplot(211)
    #     plt.plot(seq[:, seq_num, :])
    #     ax1.set_title('input')
    #     ax2 = plt.subplot(212)
    #     true_targets = plt.step(xrange(n_steps), targets[:, seq_num, :],
    #                             marker='o')

    #     guess = model.predict_proba(seq[:, seq_num, :][:, np.newaxis, :])
    #     guessed_targets = plt.step(xrange(n_steps), guess.squeeze())
    #     plt.setp(guessed_targets, linestyle='--', marker='d')
    #     for i, x in enumerate(guessed_targets):
    #         x.set_color(true_targets[i].get_color())
    #     ax2.set_ylim((-0.1, 1.1))
    #     ax2.set_title('solid: true output, dashed: model output (prob)')


# def test_softmax(n_epochs=250, optimizer='cg'):
#     """ Test RNN with softmax outputs. """
#     n_hidden = 10
#     n_in = 5
#     n_steps = 10
#     n_seq = 10  # per batch
#     n_batches = 50
#     n_classes = 3
#     n_out = n_classes  # restricted to single softmax per time step

#     np.random.seed(0)
#     # simple lag test
#     seq = np.random.randn(n_steps, n_seq * n_batches, n_in)
#     targets = np.zeros((n_steps, n_seq * n_batches), dtype=np.int)

#     thresh = 0.5
#     # if lag 1 (dim 3) is greater than lag 2 (dim 0) + thresh
#     # class 1
#     # if lag 1 (dim 3) is less than lag 2 (dim 0) - thresh
#     # class 2
#     # if lag 2(dim0) - thresh <= lag 1 (dim 3) <= lag2(dim0) + thresh
#     # class 0
#     targets[2:, :][seq[1:-1, :, 3] > seq[:-2, :, 0] + thresh] = 1
#     targets[2:, :][seq[1:-1, :, 3] < seq[:-2, :, 0] - thresh] = 2
#     #targets[:, 2:, 0] = np.cast[np.int](seq[:, 1:-1, 3] > seq[:, :-2, 0])

#     model = MetaRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
#                     learning_rate=0.005, learning_rate_decay=0.999,
#                     n_epochs=n_epochs, batch_size=n_seq, activation='tanh',
#                     output_type='softmax')

#     model.fit(seq, targets, validate_every=10, compute_zero_one=True,
#               optimizer=optimizer)

#     seqs = xrange(10)

#     plt.close('all')
#     for seq_num in seqs:
#         fig = plt.figure()
#         ax1 = plt.subplot(211)
#         plt.plot(seq[:, seq_num])
#         ax1.set_title('input')
#         ax2 = plt.subplot(212)

#         # blue line will represent true classes
#         true_targets = plt.step(xrange(n_steps), targets[:, seq_num],
#                                 marker='o')

#         # show probabilities (in b/w) output by model
#         guess = model.predict_proba(seq[:, seq_num][:, np.newaxis])
#         guessed_probs = plt.imshow(guess.squeeze().T, interpolation='nearest',
#                                    cmap='gray')
#         ax2.set_title('blue: true class, grayscale: probs assigned by model')


if __name__ == "__main__":

    # raise NotImplementedError('still working on this')

    logging.basicConfig(level=logging.INFO)
    t0 = time.time()
    # test_real()
    # problem takes more epochs to solve
    test_binary(multiple_out=False, n_epochs=100)
    # test_softmax(n_epochs=250)
    # print "Elapsed time: %f" % (time.time() - t0)
