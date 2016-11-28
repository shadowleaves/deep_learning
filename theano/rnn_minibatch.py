""" Vanilla RNN
Parallelizes scan over sequences by using mini-batches.
@author Graham Taylor
"""
import numpy as np
import theano
import theano.tensor as T
# from sklearn.base import BaseEstimator
import logging
import time
import os
import datetime
import cPickle as pickle
from collections import OrderedDict

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt
plt.ion()

mode = theano.Mode(linker='cvm')
#mode = 'DEBUG_MODE'


def xavier(shape):
    assert len(shape) == 2
    n_in, n_out = shape
    a = np.sqrt(6.0 / (n_in + n_out))
    return np.random.uniform(-a, a, (n_in, n_out))


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)


# zeros = lambda shape: np.zeros(shape)


class RNN(object):
    """    Recurrent neural network class
    Supported output types:
    real : linear output units, use mean-squared error
    binary : binary output units, use cross-entropy error
    softmax : single softmax out, use cross-entropy error
    """

    def __init__(self, input, n_in, n_hidden, n_out, activation=T.tanh,
                 output_type='real'):

        self.input = input
        self.activation = activation
        self.output_type = output_type

        self.batch_size = T.iscalar()

        # theta is a vector of all trainable parameters
        # it represents the value of W, W_in, W_out, h0, bh, by
        theta_shape = n_hidden ** 2 + n_in * n_hidden + n_hidden * n_out + \
            n_hidden + n_hidden + n_out
        self.theta = theano.shared(value=np.zeros(theta_shape,
                                                  dtype=theano.config.floatX))

        # Parameters are reshaped views of theta
        param_idx = 0  # pointer to somewhere along parameter vector

        # recurrent weights as a shared variable
        self.W = self.theta[param_idx:(param_idx + n_hidden ** 2)].reshape(
            (n_hidden, n_hidden))
        self.W.name = 'W'
        W_init = xavier((n_hidden, n_hidden))
        param_idx += n_hidden ** 2

        # input to hidden layer weights
        self.W_in = self.theta[param_idx:(param_idx + n_in *
                                          n_hidden)].reshape((n_in, n_hidden))
        self.W_in.name = 'W_in'
        W_in_init = xavier((n_in, n_hidden))
        param_idx += n_in * n_hidden

        # hidden to output layer weights
        self.W_out = self.theta[param_idx:(param_idx + n_hidden *
                                           n_out)].reshape((n_hidden, n_out))
        self.W_out.name = 'W_out'

        W_out_init = xavier((n_hidden, n_out))
        param_idx += n_hidden * n_out

        self.h0 = self.theta[param_idx:(param_idx + n_hidden)]
        self.h0.name = 'h0'
        h0_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        param_idx += n_hidden

        self.bh = self.theta[param_idx:(param_idx + n_hidden)]
        self.bh.name = 'bh'
        bh_init = np.zeros((n_hidden,), dtype=theano.config.floatX)
        param_idx += n_hidden

        self.by = self.theta[param_idx:(param_idx + n_out)]
        self.by.name = 'by'
        by_init = np.zeros((n_out,), dtype=theano.config.floatX)
        param_idx += n_out

        assert(param_idx == theta_shape)

        # for convenience
        self.params = [self.W, self.W_in, self.W_out, self.h0, self.bh,
                       self.by]

        # shortcut to norms (for monitoring)
        self.l2_norms = {}
        for param in self.params:
            self.l2_norms[param] = T.sqrt(T.sum(param ** 2))

        # initialize parameters
        # DEBUG_MODE gives division by zero error when we leave parameters
        # as zeros
        self.theta.set_value(
            np.concatenate([x.ravel() for x in
                            (
                W_init, W_in_init, W_out_init, h0_init, bh_init, by_init)]))

        self.theta_update = theano.shared(
            value=np.zeros(theta_shape, dtype=theano.config.floatX))

        # recurrent function (using tanh activation function) and arbitrary output
        # activation function
        def step(x_t, h_tm1):
            h_t = self.activation(T.dot(x_t, self.W_in) +
                                  T.dot(h_tm1, self.W) + self.bh)
            y_t = T.dot(h_t, self.W_out) + self.by
            return h_t, y_t

        # the hidden state `h` for the entire sequence, and the output for the
        # entire sequence `y` (first dimension is always time)
        # Note the implementation of weight-sharing h0 across variable-size
        # batches using T.ones multiplying h0
        # Alternatively, T.alloc approach is more robust
        [self.h, self.y_pred], _ = theano.scan(
            step,
            sequences=self.input,
            outputs_info=[T.alloc(self.h0, self.input.shape[1],
                                  n_hidden), None])
        # outputs_info=[T.ones(shape=(self.input.shape[1],
        # self.h0.shape[0])) * self.h0, None])

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = 0
        self.L1 += abs(self.W.sum())
        self.L1 += abs(self.W_in.sum())
        self.L1 += abs(self.W_out.sum())

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = 0
        self.L2_sqr += (self.W ** 2).sum()
        self.L2_sqr += (self.W_in ** 2).sum()
        self.L2_sqr += (self.W_out ** 2).sum()

        if self.output_type == 'real':
            self.loss = lambda y: self.mse(y)
        elif self.output_type == 'binary':
            # push through sigmoid
            self.p_y_given_x = T.nnet.sigmoid(self.y_pred)  # apply sigmoid
            self.y_out = T.round(self.p_y_given_x)  # round to {0,1}
            self.loss = lambda y: self.nll_binary(y)
        elif self.output_type == 'softmax':
            # push through softmax, computing vector of class-membership
            # probabilities in symbolic form
            #
            # T.nnet.softmax will not operate on T.tensor3 types, only matrices
            # We take our n_steps x n_seq x n_classes output from the net
            # and reshape it into a (n_steps * n_seq) x n_classes matrix
            # apply softmax, then reshape back
            y_p = self.y_pred
            y_p_m = T.reshape(y_p, (y_p.shape[0] * y_p.shape[1], -1))
            y_p_s = T.nnet.softmax(y_p_m)
            self.p_y_given_x = T.reshape(y_p_s, y_p.shape)

            # compute prediction as class whose probability is maximal
            self.y_out = T.argmax(self.p_y_given_x, axis=-1)
            self.loss = lambda y: self.nll_multiclass(y)

        else:
            raise NotImplementedError

    def mse(self, y):
        # error between output and target
        return T.mean((self.y_pred - y) ** 2)

    def nll_binary(self, y):
        # negative log likelihood based on binary cross entropy error
        return T.mean(T.nnet.binary_crossentropy(self.p_y_given_x, y))

    def nll_multiclass(self, y):
        # negative log likelihood based on multiclass cross entropy error
        #
        # Theano's advanced indexing is limited
        # therefore we reshape our n_steps x n_seq x n_classes tensor3 of probs
        # to a (n_steps * n_seq) x n_classes matrix of probs
        # so that we can use advanced indexing (i.e. get the probs which
        # correspond to the true class)
        # the labels y also must be flattened when we do this to use the
        # advanced indexing
        p_y = self.p_y_given_x
        p_y_m = T.reshape(p_y, (p_y.shape[0] * p_y.shape[1], -1))
        y_f = y.flatten(ndim=1)
        return -T.mean(T.log(p_y_m)[T.arange(p_y_m.shape[0]), y_f])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch
        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_out.ndim:
            raise TypeError('y should have the same shape as self.y_out',
                            ('y', y.type, 'y_out', self.y_out.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_out, y))
        else:
            raise NotImplementedError()


class MetaRNN(object):
    def __init__(self, n_in=5, n_hidden=50, n_out=5, learning_rate=0.01,
                 n_epochs=100, batch_size=100, L1_reg=0.00, L2_reg=0.00,
                 learning_rate_decay=1,
                 activation='tanh', output_type='real', final_momentum=0.9,
                 initial_momentum=0.5, momentum_switchover=5,
                 snapshot_every=None, snapshot_path='/tmp'):
        self.n_in = int(n_in)
        self.n_hidden = int(n_hidden)
        self.n_out = int(n_out)
        self.learning_rate = float(learning_rate)
        self.learning_rate_decay = float(learning_rate_decay)
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.L1_reg = float(L1_reg)
        self.L2_reg = float(L2_reg)
        self.activation = activation
        self.output_type = output_type
        self.initial_momentum = float(initial_momentum)
        self.final_momentum = float(final_momentum)
        self.momentum_switchover = int(momentum_switchover)
        if snapshot_every is not None:
            self.snapshot_every = int(snapshot_every)
        else:
            self.snapshot_every = None
        self.snapshot_path = snapshot_path

        self.ready()

    def ready(self):
        # input (where first dimension is time)
        self.x = T.tensor3(name='x')
        # target (where first dimension is time)
        if self.output_type == 'real':
            self.y = T.tensor3(name='y', dtype=theano.config.floatX)
        elif self.output_type == 'binary':
            self.y = T.tensor3(name='y', dtype='int32')
        elif self.output_type == 'softmax':  # now it is a matrix (T x n_seq)
            self.y = T.matrix(name='y', dtype='int32')
        else:
            raise NotImplementedError

        # learning rate
        self.lr = T.scalar()

        if self.activation == 'tanh':
            activation = T.tanh
        elif self.activation == 'sigmoid':
            activation = T.nnet.sigmoid
        elif self.activation == 'relu':
            activation = lambda x: x * (x > 0)
        elif self.activation == 'cappedrelu':
            activation = lambda x: T.minimum(x * (x > 0), 6)
        else:
            raise NotImplementedError

        self.rnn = RNN(input=self.x, n_in=self.n_in,
                       n_hidden=self.n_hidden, n_out=self.n_out,
                       activation=activation, output_type=self.output_type)

        if self.output_type == 'real':
            self.predict = theano.function(inputs=[self.x, ],
                                           outputs=self.rnn.y_pred,
                                           mode=mode)
        elif self.output_type == 'binary':
            self.predict_proba = theano.function(
                inputs=[self.x, ],
                outputs=self.rnn.p_y_given_x, mode=mode)
            self.predict = theano.function(
                inputs=[self.x, ],
                outputs=T.round(
                    self.rnn.p_y_given_x),
                mode=mode)
        elif self.output_type == 'softmax':
            self.predict_proba = theano.function(
                inputs=[self.x, ],
                outputs=self.rnn.p_y_given_x, mode=mode)
            self.predict = theano.function(inputs=[self.x, ],
                                           outputs=self.rnn.y_out, mode=mode)
        else:
            raise NotImplementedError

    def shared_dataset(self, data_xy, borrow=True):
        """ Load the dataset into shared variables """

        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=True)

        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=True)

        if self.output_type in ('binary', 'softmax'):
            return shared_x, T.cast(shared_y, 'int32')
        else:
            return shared_x, shared_y

    def __getstate__(self):
        """ Return state sequence."""
        params = self._get_params()  # parameters set in constructor
        theta = self.rnn.theta.get_value()
        state = (params, theta)
        return state

    def _set_weights(self, theta):
        """ Set fittable parameters from weights sequence.
        """
        self.rnn.theta.set_value(theta)

    def __setstate__(self, state):
        """ Set parameters from state sequence.
        """
        params, theta = state
        self.set_params(**params)
        self.ready()
        self._set_weights(theta)

    def save(self, fpath='.', fname=None):
        """ Save a pickled representation of Model state. """
        fpathstart, fpathext = os.path.splitext(fpath)
        if fpathext == '.pkl':
            # User supplied an absolute path to a pickle file
            fpath, fname = os.path.split(fpath)

        elif fname is None:
            # Generate filename based on date
            date_obj = datetime.datetime.now()
            date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
            class_name = self.__class__.__name__
            fname = '%s.%s.pkl' % (class_name, date_str)

        fabspath = os.path.join(fpath, fname)

        logger.info("Saving to %s ..." % fabspath)
        file = open(fabspath, 'wb')
        state = self.__getstate__()
        pickle.dump(state, file, protocol=pickle.HIGHEST_PROTOCOL)
        file.close()

    def load(self, path):
        """ Load model parameters from path. """
        logger.info("Loading from %s ..." % path)
        file = open(path, 'rb')
        state = pickle.load(file)
        self.__setstate__(state)
        file.close()

    def optional_output(self, train_set_x, show_norms=True, show_output=True):
        """ Produces some debugging output. """
        if show_norms:
            norm_output = []
            for param in self.rnn.params:
                norm_output.append('%s: %6.4f' % (param.name,
                                                  self.get_norms[param]()))
            logger.info("norms: {" + ', '.join(norm_output) + "}")

        if show_output:
            # show output for a single case
            if self.output_type == 'binary':
                output_fn = self.predict_proba
            else:
                output_fn = self.predict
            logger.info(
                "sample output: " +
                str(output_fn(train_set_x.get_value(
                    borrow=True)[:, 0, :][:, np.newaxis, :]).flatten()))

    def rmsprop(lr, tparams, grads, x, mask, y, cost):
        """
        A variant of  SGD that scales the step size by running average of the
        recent step norms.

        Parameters
        ----------
        lr : Theano SharedVariable
            Initial learning rate
        tpramas: Theano SharedVariable
            Model parameters
        grads: Theano variable
            Gradients of cost w.r.t to parameres
        x: Theano variable
            Model inputs
        mask: Theano variable
            Sequence mask
        y: Theano variable
            Targets
        cost: Theano variable
            Objective fucntion to minimize

        Notes
        -----
        For more information, see [Hint2014]_.

        .. [Hint2014] Geoff Hinton, *Neural Networks for Machine Learning*,
           lecture 6a,
           http://cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf
        """

        zipped_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                      name='%s_grad' % k)
                        for k, p in tparams.items()]
        running_grads = [theano.shared(p.get_value() * numpy_floatX(0.),
                                       name='%s_rgrad' % k)
                         for k, p in tparams.items()]
        running_grads2 = [theano.shared(p.get_value() * numpy_floatX(0.),
                                        name='%s_rgrad2' % k)
                          for k, p in tparams.items()]

        zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
        rgup = [(rg, 0.95 * rg + 0.05 * g)
                for rg, g in zip(running_grads, grads)]
        rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
                 for rg2, g in zip(running_grads2, grads)]

        f_grad_shared = theano.function([x, mask, y], cost,
                                        updates=zgup + rgup + rg2up,
                                        name='rmsprop_f_grad_shared')

        updir = [theano.shared(p.get_value() * numpy_floatX(0.),
                               name='%s_updir' % k)
                 for k, p in tparams.items()]
        updir_new = [(ud, 0.9 * ud - 1e-4 * zg / tensor.sqrt(rg2 - rg ** 2 + 1e-4))
                     for ud, zg, rg, rg2 in zip(updir, zipped_grads, running_grads,
                                                running_grads2)]
        param_up = [(p, p + udn[1])
                    for p, udn in zip(tparams.values(), updir_new)]
        f_update = theano.function([lr], [], updates=updir_new + param_up,
                                   on_unused_input='ignore',
                                   name='rmsprop_f_update')

        return f_grad_shared, f_update

    def fit(self, X_train, Y_train, X_test=None, Y_test=None,
            validate_every=100, optimizer='sgd', compute_zero_one=False,
            show_norms=True, show_output=True):
        """ Fit model
        Pass in X_test, Y_test to compute test error and report during
        training.
        X_train : ndarray (T x n_in)
        Y_train : ndarray (T x n_out)
        validation_frequency : int
            in terms of number of epochs
        optimizer : string
            Optimizer type.
            Possible values:
                'sgd'  : batch stochastic gradient descent
                'cg'   : nonlinear conjugate gradient algorithm
                         (scipy.optimize.fmin_cg)
                'bfgs' : quasi-Newton method of Broyden, Fletcher, Goldfarb,
                         and Shanno (scipy.optimize.fmin_bfgs)
                'l_bfgs_b' : Limited-memory BFGS (scipy.optimize.fmin_l_bfgs_b)
        compute_zero_one : bool
            in the case of binary output, compute zero-one error in addition to
            cross-entropy error
        show_norms : bool
            Show L2 norms of individual parameter groups while training.
        show_output : bool
            Show the model output on first training case while training.
        """
        if X_test is not None:
            assert(Y_test is not None)
            self.interactive = True
            test_set_x, test_set_y = self.shared_dataset((X_test, Y_test))
        else:
            self.interactive = False

        train_set_x, train_set_y = self.shared_dataset((X_train, Y_train))

        if compute_zero_one:
            assert(self.output_type == 'binary' or
                   self.output_type == 'softmax')
        # compute number of minibatches for training
        # note that cases are the second dimension, not the first
        n_train = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches = int(np.ceil(1.0 * n_train / self.batch_size))
        if self.interactive:
            n_test = test_set_x.get_value(borrow=True).shape[0]
            n_test_batches = int(np.ceil(1.0 * n_test / self.batch_size))

        # validate_every is specified in terms of epochs
        validation_frequency = validate_every * n_train_batches

        ######################
        # BUILD ACTUAL MODEL #
        ######################
        logger.info('... building the model')

        index = T.lscalar('index')    # index to a [mini]batch
        n_ex = T.lscalar('n_ex')      # total number of examples
        # learning rate (may change)
        l_r = T.scalar('l_r', dtype=theano.config.floatX)
        mom = T.scalar('mom', dtype=theano.config.floatX)  # momentum

        cost = self.rnn.loss(self.y) \
            + self.L1_reg * self.rnn.L1 \
            + self.L2_reg * self.rnn.L2_sqr

        # Proper implementation of variable-batch size evaluation
        # Note that classifier.errors() returns the mean error
        # But the last batch may be a smaller size
        # So we keep around the effective_batch_size (whose last element may
        # be smaller than the rest)
        # And weight the reported error by the batch_size when we average
        # Also, by keeping batch_start and batch_stop as symbolic variables,
        # we make the theano function easier to read
        batch_start = index * self.batch_size
        batch_stop = T.minimum(n_ex, (index + 1) * self.batch_size)
        effective_batch_size = batch_stop - batch_start

        get_batch_size = theano.function(
            inputs=[index, n_ex],
            outputs=effective_batch_size)

        compute_train_error = theano.function(
            inputs=[index, n_ex],
            outputs=self.rnn.loss(self.y),
            givens={self.x: train_set_x[batch_start:batch_stop, :],
                    self.y: train_set_y[batch_start:batch_stop, :]},
            mode=mode)

        if compute_zero_one:
            compute_train_zo = theano.function(
                inputs=[index, n_ex],
                outputs=self.rnn.errors(self.y),
                givens={self.x: train_set_x[batch_start:batch_stop, :],
                        self.y: train_set_y[batch_start:batch_stop, :]},
                mode=mode)

        if self.interactive:
            compute_test_error = theano.function(
                inputs=[index, n_ex],
                outputs=self.rnn.loss(self.y),
                givens={self.x: test_set_x[batch_start:batch_stop, :],
                        self.y: test_set_y[batch_start:batch_stop, :]},
                mode=mode)

            if compute_zero_one:
                compute_test_zo = theano.function(
                    inputs=[index, n_ex],
                    outputs=self.rnn.errors(
                        self.y),
                    givens={self.x: test_set_x[batch_start:batch_stop, :],
                            self.y: test_set_y[batch_start:batch_stop, :]},
                    mode=mode)

        self.get_norms = {}
        for param in self.rnn.params:
            self.get_norms[param] = theano.function(
                inputs=[],
                outputs=self.rnn.l2_norms[param], mode=mode)

        # compute the gradient of cost with respect to theta using BPTT
        gtheta = T.grad(cost, self.rnn.theta)

        if optimizer == 'sgd':

            updates = OrderedDict()
            theta = self.rnn.theta
            theta_update = self.rnn.theta_update
            # careful here, update to the shared variable
            # cannot depend on an updated other shared variable
            # since updates happen in parallel
            # so we need to be explicit
            upd = mom * theta_update - l_r * gtheta
            updates[theta_update] = upd
            updates[theta] = theta + upd

            # compiling a Theano function `train_model` that returns the
            # cost, but in the same time updates the parameter of the
            # model based on the rules defined in `updates`
            train_model = theano.function(
                inputs=[index, n_ex, l_r, mom],
                outputs=cost,
                updates=updates,
                givens={self.x: train_set_x[batch_start:batch_stop, :],
                        self.y: train_set_y[batch_start:batch_stop, :]},
                mode=mode)

            ###############
            # TRAIN MODEL #
            ###############
            logger.info('... training')
            epoch = 0

            while (epoch < self.n_epochs):
                epoch = epoch + 1
                effective_momentum = self.final_momentum \
                    if epoch > self.momentum_switchover \
                    else self.initial_momentum

                for minibatch_idx in xrange(n_train_batches):
                    minibatch_avg_cost = train_model(
                        minibatch_idx, n_train,
                        self.learning_rate,
                        effective_momentum)

                    # iteration number (how many weight updates have we made?)
                    # epoch is 1-based, index is 0 based
                    iter = (epoch - 1) * n_train_batches + minibatch_idx + 1

                    if iter % validation_frequency == 0:
                        # compute loss on training set
                        train_losses = [compute_train_error(i, n_train)
                                        for i in xrange(n_train_batches)]
                        train_batch_sizes = [get_batch_size(i, n_train)
                                             for i in xrange(n_train_batches)]

                        this_train_loss = np.average(train_losses,
                                                     weights=train_batch_sizes)

                        if compute_zero_one:
                            train_zero_one = [compute_train_zo(i, n_train)
                                              for i in xrange(n_train_batches)]

                            this_train_zero_one = np.average(
                                train_zero_one,
                                weights=train_batch_sizes)

                        if self.interactive:
                            test_losses = [
                                compute_test_error(i, n_test)
                                for i in xrange(n_test_batches)]

                            test_batch_sizes = [
                                get_batch_size(i, n_test)
                                for i in xrange(n_test_batches)]

                            this_test_loss = np.average(
                                test_losses,
                                weights=test_batch_sizes)

                            if compute_zero_one:
                                test_zero_one = [
                                    compute_test_zo(i, n_test)
                                    for i in xrange(n_test_batches)]

                                this_test_zero_one = np.average(
                                    test_zero_one,
                                    weights=test_batch_sizes)

                            if compute_zero_one:
                                logger.info(
                                    'epoch %i, mb %i/%i, tr loss %f, '
                                    'tr zo %f, te loss %f '
                                    'te zo %f lr: %f' %
                                    (epoch, minibatch_idx + 1,
                                     n_train_batches,
                                     this_train_loss, this_train_zero_one,
                                     this_test_loss, this_test_zero_one,
                                     self.learning_rate))
                            else:
                                logger.info(
                                    'epoch %i, mb %i/%i, tr loss %f '
                                    'te loss %f lr: %f' %
                                    (epoch, minibatch_idx + 1, n_train_batches,
                                     this_train_loss, this_test_loss,
                                     self.learning_rate))

                        else:
                            if compute_zero_one:
                                logger.info(
                                    'epoch %i, mb %i/%i, train loss %f'
                                    ' train zo %f '
                                    'lr: %f' % (epoch,
                                                minibatch_idx + 1,
                                                n_train_batches,
                                                this_train_loss,
                                                this_train_zero_one,
                                                self.learning_rate))
                            else:
                                logger.info(
                                    'epoch %i, mb %i/%i, train loss %f'
                                    ' lr: %f' % (epoch,
                                                 minibatch_idx + 1,
                                                 n_train_batches,
                                                 this_train_loss,
                                                 self.learning_rate))

                        self.optional_output(train_set_x, show_norms,
                                             show_output)

                self.learning_rate *= self.learning_rate_decay

                if self.snapshot_every is not None:
                    if (epoch + 1) % self.snapshot_every == 0:
                        date_obj = datetime.datetime.now()
                        date_str = date_obj.strftime('%Y-%m-%d-%H:%M:%S')
                        class_name = self.__class__.__name__
                        fname = '%s.%s-snapshot-%d.pkl' % (class_name,
                                                           date_str, epoch + 1)
                        fabspath = os.path.join(self.snapshot_path, fname)
                        self.save(fpath=fabspath)

            ###############
            # TRAIN MODEL #
            ###############
            logger.info('... training')
            # using scipy conjugate gradient optimizer
            # import scipy.optimize
            # logger.info("Optimizing using %s..." % of.__name__)
            start_time = time.clock()

            # keep track of epochs externally
            # these get updated through callback
            self.epoch = 0

            end_time = time.clock()

            print "Optimization time: %f" % (end_time - start_time)

        else:
            raise NotImplementedError
