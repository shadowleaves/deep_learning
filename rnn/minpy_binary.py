#!/usr/bin/env python

""" algo based on http://peterroelants.github.io/posts/rnn_implementation_part02/
skeleton code from MinPy (https://github.com/dmlc/minpy)
by shadowleaves 2016
"""

# Python imports
# import itertools
import minpy.numpy as mp
import numpy as np  # Matrix and vector computation package
# import matplotlib
# import matplotlib.pyplot as plt  # Plotting library
# Allow matplotlib to plot inside this notebook
# %matplotlib inline
# Set the seed of the numpy random number generator so that the tutorial
# is reproducable

# sys.path.append('/home/hyan/codebase/minpy/minpy')
# from minpy.context import set_context, gpu
# set_context(gpu(0))  # set the global context as gpu(0)

from minpy.nn import layers
from minpy.nn.model import ModelBase
from minpy.nn.solver import Solver
from minpy.nn.io import NDArrayIter
from minpy import core

# from minpy.core import convert_args
# from numpy_binary import create_dataset, printSample

from utils.timedate import timing


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

# @convert_args


def rmsprop_mom(x, dx, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared gradient
    values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
                  gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None:
        config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('momentum', 0.9)
    config.setdefault('cache', np.zeros_like(x))
    v = config.get('velocity', np.zeros_like(x))

    cache = config['cache']
    cache = cache * config['decay_rate'] + dx**2 * (1 - config['decay_rate'])

    if True:
        grad_norm = dx / (mp.sqrt(cache) + config['epsilon'])
        v = v * config['momentum'] - grad_norm * config['learning_rate']
        next_x = x + v
        config['velocity'] = v

    # next_x = x + v
    else:
        next_x = x - config['learning_rate'] * dx / (mp.sqrt(cache) +
                                                     config['epsilon'])

    config['cache'] = cache
    return next_x, config


class CustomSolver(Solver):
    """custom solver for binary addition"""

    def __init__(self, *kargs, **kwargs):
        super(CustomSolver, self).__init__(*kargs, **kwargs)
        self.update_rule = rmsprop_mom

    # def _step(self, batch):
    #     """
    #     Make a single gradient update. This is called by train() and should not
    #     be called manually.
    #     """
    #     # Compute loss and gradient
    #     def loss_func(*params):
    #         # It seems that params are not used in forward function. But since we will pass
    #         # model.params as arguments, we are ok here.
    #         predict = self.model.forward_batch(batch, mode='train')
    #         return self.model.loss_batch(batch, predict)

    #     param_arrays = list(self.model.params.values())
    #     param_keys = list(self.model.params.keys())
    #     grad_and_loss_func = core.grad_and_loss(
    #         loss_func, argnum=range(len(param_arrays)))

    #     t0 = timing()
    #     grad_arrays, loss = grad_and_loss_func(*param_arrays)
    #     timing(t0, 'grad', 'ms')
    #     grads = dict(zip(param_keys, grad_arrays))

    #     self.loss_history.append(loss.asnumpy())

    #     # Perform a parameter update
    #     for p, w in self.model.params.items():
    #         dw = grads[p]
    #         config = self.optim_configs[p]
    #         next_w, next_config = self.update_rule(w, dw, config)
    #         self.model.params[p] = next_w
    #         self.optim_configs[p] = next_config

    def _reset(self):
        """
        Set up some book-keeping variables for optimization. Don't call this
        manually.
        """
        # Set up some variables for book-keeping
        self.epoch = 0
        self.best_val_acc = 0
        self.best_params = {}
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []
        self._reset_data_iterators()

        # Make a deep copy of the optim_config for each parameter
        self.optim_configs = {}
        for p in self.model.param_configs:
            d = {k: v for k, v in self.optim_config.items()}
            self.optim_configs[p] = d
        # Overwrite it if the model specify the rules

        # Make a deep copy of the init_config for each parameter
        # and set each param to their own init_rule and init_config
        self.init_rules = {}
        self.init_configs = {}
        for p in self.model.param_configs:
            if 'init_rule' in self.model.param_configs[p]:
                init_rule = self.model.param_configs[p]['init_rule']
                init_config = self.model.param_configs[p].get('init_config',
                                                              {})
            else:
                init_rule = self.init_rule
                init_config = {k: v for k, v in self.init_config.items()}
            # replace string name with actual function
            if not hasattr(init_rule, '__call__'):
                if not hasattr(init, init_rule):
                    raise ValueError('Invalid init_rule "%s"' % init_rule)
                init_rule = getattr(init, init_rule)
            self.init_rules[p] = init_rule
            self.init_configs[p] = init_config

    def check_accuracy(self, dataiter, num_samples=None):
        """
        Check accuracy of the model on the provided data.
        Inputs:
        - dataiter: data iterator that can produce batches.
        - num_samples: If not None and dataiter has more than num_samples datapoints,
          subsample the data and only test the model on num_samples datapoints.
        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """

        # Maybe subsample the data
        N = dataiter.num_data
        check_dataiter = dataiter
        if num_samples is not None and N > num_samples:
            # Sample a sub iter
            check_dataiter = dataiter.getsubiter(num_samples)
        else:
            # Use the entire dataiter otherwise.
            check_dataiter.reset()

        acc_count = 0
        num_samples = 0
        for each_batch in check_dataiter:
            predict = self.model.forward_batch(
                each_batch, mode='test').asnumpy()
            # TODO(minjie): multiple labels.

            diff = mp.round(predict) - each_batch.label[0]
            diff = mp.sum(mp.abs(diff), axis=1)
            acc_count += mp.sum(diff == 0)
            # print acc_count
            # acc_count += np.sum(
            #     np.argmax(predict, axis=1) == each_batch.label[0])
            num_samples += check_dataiter.batch_size

        # if self.epoch > 40:
        #     import pdb
        #     pdb.set_trace()

        return acc_count / num_samples


def xavier(shape, config):
    assert len(shape) == 2
    n_in, n_out = shape
    a = np.sqrt(6.0 / (n_in + n_out))
    return mp.random.uniform(-a, a, (n_in, n_out))

zeros = lambda shape, config: mp.zeros(shape)


class RNNModel(ModelBase):
    """RNN to perform binary addition of 2 numbers."""

    def __init__(self,  # batch_size=100,
                 input_size=2,
                 hidden_size=3,
                 num_classes=1,
                 # nb_of_outputs,
                 # nb_of_states, sequence_len
                 ):
        """Initialse the network layers."""
        super(RNNModel, self).__init__()
        # input layer
        self.add_param(name='Wx', shape=(input_size, hidden_size),
                       init_rule=xavier
                       )
        # recurrent layer
        self.add_param(name='Wh', shape=(hidden_size, hidden_size),
                       init_rule=xavier
                       )
        # recurrent bias
        self.add_param(name='b', shape=(hidden_size, ),
                       init_rule=zeros
                       )
        # output layer
        self.add_param(name='Wa', shape=(hidden_size, num_classes),
                       init_rule=xavier
                       )
        # output bias
        self.add_param(name='ba', shape=(num_classes, ),
                       init_rule=zeros
                       )

    def forward(self, X, mode):
        # seq_len = X.shape[1]
        batch_size = X.shape[0]
        hidden_size = self.params['Wh'].shape[0]
        class_size = self.params['Wa'].shape[1]
        h = mp.zeros((batch_size, hidden_size))
        y = mp.zeros((batch_size, 0, class_size))

        N, T, D = X.shape

        for t in xrange(T):
            h = layers.rnn_step(X[:, t, :], h, self.params['Wx'],
                                self.params['Wh'], self.params['b'])
            z = layers.affine(h, self.params['Wa'], self.params['ba'])
            y_step = layers.sigmoid(z).reshape(N, 1, 1)
            y = mp.append(y, y_step, axis=1)

        return y

    def loss(self, Y, T, ep=1e-10):
        # import pdb
        # pdb.set_trace()
        loss = -mp.sum(mp.multiply(T, mp.log(Y + ep)) +
                       mp.multiply((1 - T), mp.log(1 - Y + ep))) \
            / (Y.shape[0] * Y.shape[1])

        return loss
        # return layers.temporal_softmax_loss(Y, T, mask)
        # return layers.softmax_loss(predict, y)


if __name__ == '__main__':

    # import matplotlib
    # import matplotlib.pyplot as plt  # Plotting library

    # Create dataset
    nb_train = 2000  # Number of training samples
    nb_test = 100
    # Addition of 2 n-bit numbers can result in a n+1 bit number
    sequence_len = 7  # Length of the binary sequence

    # Create training samples

    np.random.seed(seed=2)
    X_train, T_train = create_dataset(nb_train, sequence_len)
    X_test, T_test = create_dataset(nb_test, sequence_len)
    print('X_train shape: {0}'.format(X_train.shape))
    print('T_train shape: {0}'.format(T_train.shape))

    # Print the first sample
    printSample(X_train[0, :, 0], X_train[0, :, 1], T_train[0, :, :])

    # Set hyper-parameters
    lmbd = 0.5  # Rmsprop lambda
    learning_rate = 0.05  # Learning rate
    momentum_term = 0.80  # Momentum term
    eps = 1e-6  # Numerical stability term to prevent division by zero
    mb_size = 100  # Size of the minibatches (number of samples)

    # Create the network
    # nb_of_states = 4  # Number of states in the recurrent layer
    model = RNNModel()
    # Set the initial parameters
    # Number of parameters in the network

    train_dataiter = NDArrayIter(data=X_train,
                                 label=T_train,
                                 batch_size=mb_size,
                                 shuffle=True)
    test_dataiter = NDArrayIter(data=X_test,
                                label=T_test,
                                batch_size=mb_size,
                                shuffle=False)

    # Create solver.
    solver = CustomSolver(model,
                          train_dataiter,
                          test_dataiter,
                          num_epochs=20,
                          init_rule='custom',
                          # init_config={'function': init},
                          update_rule='rmsprop',
                          optim_config={
                              'learning_rate': 0.05,
                              'momentum': 0.8,
                              'epsilon': 1e-6,
                              'decay_rate': 0.5,
                          },
                          verbose=True,
                          print_every=10)

    solver.init()
    solver.train()
    # solver.validate()
    # Create test samples
    if True:
        # Push test data through network
        for batch in test_dataiter:
            Y = model.forward_batch(batch, mode='test')
            # Yf = model.getOutput(X_test)
            Y = mp.round(Y)
            X = batch.data[0]
            T = batch.label[0]
            # Print out all test examples
            for i in range(X.shape[0]):
                printSample(X[i, :, 0], X[i, :, 1],
                            T[i, :, :], Y[i, :, :])

                if i > 5:
                    break
            break
            # print ''


'''
MIT License

Copyright (c) 2016 shadowleaves

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
