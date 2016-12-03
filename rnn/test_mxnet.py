#!/usr/bin/env python

import mxnet as mx
import numpy as np
from collections import namedtuple


RNNState = namedtuple("RNNState", ["h"])
RNNParam = namedtuple("RNNParam", ["i2h_weight", "i2h_bias",
                                   "h2h_weight", "h2h_bias"])

# RNNModel = namedtuple("RNNModel", ["rnn_exec", "symbol",
#                                    "init_states", "last_states",
#                                    "seq_data", "seq_labels", "seq_outputs",
#                                    "param_blocks"])


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
    def __init__(self, n_samples, batch_size, num_inputs, init_states):
        super(DataIter, self).__init__()
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.num_inputs = num_inputs
        self.init_states = init_states
        self.init_state_names = [x[0] for x in self.init_states]
        self.init_state_arrays = [mx.nd.zeros(x[1]) for x in init_states]

        self.provide_data = [
            ('data', (batch_size, num_inputs))] + init_states
        self.provide_label = [('label', (batch_size, ))]

    def __iter__(self):
        # print 'begin'
        # batch_data = []
        # batch_label = []
        # batch_label_weight = []
        for i in range(self.n_samples / self.batch_size):

            batch_data = np.random.sample((self.batch_size, self.num_inputs))
            batch_label = np.random.sample((self.batch_size, ))
            # batch_label_weight.append(label_weight)
            data_all = [mx.nd.array(batch_data)] + self.init_state_arrays
            label_all = [mx.nd.array(batch_label), ]

            data_names = ['data'] + self.init_state_names
            label_names = ['label']
            # batch_data = []
            # batch_label = []
            # import pdb
            # pdb.set_trace()
            yield SimpleBatch(data_names, data_all, label_names, label_all)

    def reset(self):
        pass


if __name__ == '__main__':

    nb_train = 2000  # Number of training samples
    batch_size = 100
    num_hidden = 3
    num_inputs = 2

    # X_train = np.random.sample((nb_train, num_inputs))
    # T_train = np.random.sample((nb_train, seq_len, num_inputs))

    # train_dataiter = mx.io.NDArrayIter(data=X_train,
    #                                    label=T_train,
    #                                    batch_size=batch_size,
    #                                    label_name='softmax_label',
    #                                    shuffle=True)

    init_states = [('init_h', (batch_size, num_hidden)), ]
    data_train = DataIter(nb_train, batch_size, num_inputs,
                          init_states)
    data_eval = DataIter(500, batch_size, num_inputs,
                         init_states)

    input_ = mx.symbol.Variable('data')
    wx = mx.sym.Variable('wx', shape=(num_hidden, num_inputs))
    wh = mx.sym.Variable('wh', shape=(num_hidden, num_hidden))
    b = mx.sym.Variable('b', shape=(num_hidden,))
    i2h = mx.symbol.FullyConnected(data=input_, num_hidden=num_hidden, weight=wx,
                                   no_bias=True)

    state = mx.sym.Variable('init_h', shape=(num_hidden, ))
    h2h = mx.symbol.FullyConnected(data=state, num_hidden=num_hidden, weight=wh,
                                   bias=b)

    act = mx.sym.Activation(data=i2h + h2h, act_type="tanh")
    net = mx.symbol.SoftmaxOutput(data=act,
                                  label=mx.sym.Variable('label'))

    mod = mx.mod.Module(net)
    # mod.bind(data_shapes=data_train.provide_data,
    #          label_shapes=data_train.provide_label)
    # mod.init_params(initializer=mx.initializer.Xavier)
    # mod.forward(data_train.next())
    # raise SystemExit

    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    mod.fit(data_train,
            eval_data=data_eval,
            optimizer='sgd',
            initializer=mx.initializer.Xavier,
            num_epoch=10)
