#!/usr/bin/env python

import mxnet as mx
import numpy as np

# from minpy_binary import create_dataset, printSample


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


def rnn_step(input_, state, wx, wh, b, n_hidden):
    # n_hidden = wh.shape[0]
    # update
    i2h = mx.sym.FullyConnected(data=input_,
                                weight=wx,
                                no_bias=True,
                                num_hidden=n_hidden,
                                name='i2h')
    # memory
    h2h = mx.sym.FullyConnected(data=state,
                                weight=wh,
                                bias=b,
                                num_hidden=n_hidden,
                                name='h2h')
    # activation
    return mx.sym.Activation(data=i2h + h2h, act_type="tanh")
    # return hidden


def rnn_unroll(seq_len, n_inputs, n_hidden, n_outputs, batch_size):

    # n_embed = n_inputs
    # we = mx.sym.Variable('we')
    wx = mx.sym.Variable('wx', shape=(n_hidden, n_inputs))
    wh = mx.sym.Variable('wh', shape=(n_hidden, n_hidden))
    b = mx.sym.Variable('b', shape=(n_hidden, ))
    wa = mx.sym.Variable('wa', shape=(n_outputs, n_hidden))
    ba = mx.sym.Variable('ba', shape=(n_outputs, ))

    state = mx.sym.Variable('state')

    data = mx.sym.Variable('data')
    # data = mx.sym.transpose(data=data, axis=[0, 1])
    label = mx.sym.Variable('softmax_label')
    x_mat = mx.sym.SliceChannel(
        data=data, num_outputs=seq_len, axis=1, squeeze_axis=False)

    res = []
    for i in xrange(seq_len):
        state = rnn_step(input_=x_mat[i],
                         state=state,
                         wx=wx,
                         wh=wh,
                         b=b,
                         n_hidden=n_hidden)

        fc = mx.sym.FullyConnected(data=state, weight=wa, bias=ba,
                                   num_hidden=n_outputs,
                                   name='pred',
                                   )
        sm = mx.sym.SoftmaxOutput(data=fc, label=label, name='softmax')
        res.append(sm)

    return mx.sym.Group(res)


def loss_func(Y, T, ep=1e-10):
    # import pdb
    # pdb.set_trace()
    loss = -np.sum(np.multiply(T, np.log(Y + ep)) +
                   np.multiply((1 - T), np.log(1 - Y + ep))) \
        / (Y.shape[0] * Y.shape[1])

    return loss


# class Model(mx.model.FeedForward):
#     def _init_params(self, input_shapes, overwrite=False):
#         super(Model, self)._init_params(input_shapes, overwrite=overwrite)


class SimpleBatch(object):
    def __init__(self, data_names, data, label_names, label):
        self.data = data
        self.label = label
        self.data_names = data_names
        self.label_names = label_names

    @property
    def provide_data(self):
        return [(n, x.shape) for n, x in zip(self.data_names, self.data)]

    @property
    def provide_label(self):
        return [(n, x.shape) for n, x in zip(self.label_names, self.label)]


class DataIter(mx.io.DataIter):
    def __init__(self, n_samples, batch_size, seq_len, num_inputs, num_label, init_states):
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.n_samples = n_samples

        self.seq_len = seq_len
        self.num_label = num_label
        self.init_states = init_states
        self.init_state_names = [x[0] for x in self.init_states]
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [
            ('data', (batch_size, seq_len, num_inputs))] + init_states
        self.provide_label = [('label', (batch_size, seq_len, num_label))]

        # generating data
        max_int = 2**(seq_len - 1)  # Maximum integer that can be added
        # Transform integer in binary format
        format_str = '{:0' + str(seq_len) + 'b}'
        X = np.zeros((n_samples, seq_len, 2))  # Input samples
        T = np.zeros((n_samples, seq_len, 1))  # Target samples
        # self.batch_id = 0

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

    def __iter__(self):
        print 'begin'
        batch_data = []
        batch_label = []
        # batch_label_weight = []
        for i in range(self.n_samples / self.batch_size):

            batch_data.append(data)
            batch_label.append(label)
            # batch_label_weight.append(label_weight)
            if len(batch_data) == self.batch_size:
                data_all = [mx.nd.array(batch_data)] + self.init_state_arrays
                label_all = [mx.nd.array(batch_label),
                             mx.nd.array(batch_label_weight)]
                data_names = ['data'] + self.init_state_names
                label_names = ['label']
                batch_data = []
                batch_label = []
                batch_label_weight = []
                import pdb
                pdb.set_trace()
                yield SimpleBatch(data_names, data_all, label_names, label_all)

    def reset(self):
        pass


if __name__ == '__main__':

    # Create dataset
    nb_train = 2000  # Number of training samples
    nb_test = 100
    # Addition of 2 n-bit numbers can result in a n+1 bit number
    seq_len = 7  # Length of the binary sequence
    batch_size = 100

    # Create training samples
    np.random.seed(seed=2)
    X_train, T_train = create_dataset(nb_train, seq_len)
    X_test, T_test = create_dataset(nb_test, seq_len)
    print('X_train shape: {0}'.format(X_train.shape))
    print('T_train shape: {0}'.format(T_train.shape))

    # Print the first sample
    printSample(X_train[0, :, 0], X_train[0, :, 1], T_train[0, :, :])

    train_dataiter = mx.io.NDArrayIter(data=X_train,
                                       label=T_train,
                                       batch_size=batch_size,
                                       label_name='softmax_label',
                                       shuffle=True)
    test_dataiter = mx.io.NDArrayIter(data=X_test,
                                      label=T_test,
                                      batch_size=batch_size,
                                      shuffle=False)

    sym = rnn_unroll(seq_len=seq_len,
                     n_inputs=2,
                     n_hidden=3,
                     n_outputs=1,
                     batch_size=batch_size)

    model = mx.model.FeedForward(  # ctx=contexts,
        symbol=sym,
        num_epoch=20,
        learning_rate=0.05,
        momentum=0.8,
    )

    data = mx.symbol.Variable('data')
    wx = mx.sym.Variable('wx', shape=(2, 3))
    net = mx.symbol.FullyConnected(data=data, num_hidden=3, weight=wx,
                                   no_bias=True)
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')

    mod = mx.mod.Module(sym)
    mod.bind(data_shapes=train_dataiter.provide_data,
             label_shapes=train_dataiter.provide_label)

    # model.fit(X=train_dataiter)  # , eval_data=test_dataiter,
    # eval_metric=mx.metric.np(loss_func),
    # )
    pass
