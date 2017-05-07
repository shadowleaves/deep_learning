#!/usr/bin/env python

import mxnet as mx
import numpy as np

from rnn import rnn_unroll, DataIter
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


def xavier(shape, coef=1.0):
    n_in, n_out = shape
    a = np.sqrt(6.0 / (n_in + n_out)) * coef
    res = mx.random.uniform(low=-a, high=a, shape=shape)
    return res


def loss_func(label, pred, ep=1e-10):
    loss = -np.sum(np.multiply(label, np.log(pred + ep)) +
                   np.multiply((1 - label), np.log(1 - pred + ep))) \
        / (pred.shape[0] * pred.shape[1])
    return loss


# class RMSProp(mx.optimizer.Optimizer):

#     def __init__(self, decay=0.95, momentum=0.9, **kwargs):
#         super(RMSProp, self).__init__(**kwargs)
#         self.decay = decay
#         self.momentum = momentum

#     def create_state(self, index, weight):
#         """Create additional optimizer state: mean, variance
#         Parameters
#         ----------
#         weight : NDArray
#             The weight data
#         """
#         return (mx.nd.zeros(weight.shape, weight.context),  # cache
#                 # mx.nd.zeros(weight.shape, weight.context),  # g
#                 mx.nd.zeros(weight.shape, weight.context))  # delta

#     def update(self, index, weight, grad, state, ep=1e-6):
#         """Update the parameters.
#         Parameters
#         ----------
#         index : int
#             An unique integer key used to index the parameters
#         weight : NDArray
#             weight ndarray
#         grad : NDArray
#             grad ndarray
#         state : NDArray or other objects returned by init_state
#             The auxiliary state used in optimization.
#         """
#         assert(isinstance(weight, mx.nd.NDArray))
#         assert(isinstance(grad, mx.nd.NDArray))
#         lr = self._get_lr(index)
#         # wd = self._get_wd(index)
#         self._update_count(index)

#         cache, delta = state
#         # grad = grad * self.rescale_grad
#         # if self.clip_gradient is not None:
#         # grad = clip(grad, -self.clip_gradient, self.clip_gradient)
#         cache[:] = (1 - self.decay) * (grad * grad) + self.decay * cache
#         # g[:] = (1 - self.decay) * grad + self.decay * g

#         grad_norm = grad / mx.nd.sqrt(cache + ep)  # + wd * weight
#         delta[:] = (self.momentum) * delta - lr * grad_norm
#         weight[:] += delta

#         # import pdb
#         # pdb.set_trace()

if __name__ == '__main__':

    # Create dataset
    nb_train = 2000  # Number of training samples
    nb_test = 100
    num_hidden = 3
    n_inputs = 2
    n_labels = 1
    # Addition of 2 n-bit numbers can result in a n+1 bit number
    seq_len = 7  # Length of the binary sequence
    batch_size = 100

    # Create training samples
    seed = 2
    np.random.seed(seed)
    mx.random.seed(seed)
    init_states = [
        ('init_h', (batch_size, num_hidden)),
        # ('wx', (num_hidden, n_inputs)),  # , num_hidden)),
        # ('wh', (num_hidden, num_hidden)),
        # ('b', (num_hidden, )),
        # ('wa', (n_labels, num_hidden)),
        # ('ba', (n_labels, )),
    ]
    # X, T = create_dataset(nb_train, seq_len)
    data_train = DataIter(nb_train, batch_size, seq_len, n_inputs,
                          n_labels, init_states,
                          create_dataset=create_dataset)
    # data_eval = DataIter(500, batch_size, n_inputs, init_states)

    wx = mx.sym.Variable('wx')
    wh = mx.sym.Variable('wh')
    b = mx.sym.Variable('b')
    wa = mx.sym.Variable('wa')
    ba = mx.sym.Variable('ba')

    sym = rnn_unroll(wx, wh, b, wa, ba,
                     seq_len=seq_len,
                     n_inputs=2,
                     num_hidden=3,
                     n_labels=1,
                     batch_size=batch_size)

    # mod = mx.mod.Module(sym)
    # from sys import platform
    # ctx = mx.context.gpu(0) if platform == 'darwin' else mx.context.cpu(0)

    arg_params = {
        'init_h': mx.nd.zeros((batch_size, num_hidden)),
        'wx': xavier((num_hidden, n_inputs)),
        'wh': xavier((num_hidden, num_hidden)),
        'b': mx.nd.zeros((num_hidden, )),
        'wa': xavier((n_labels, num_hidden)),
        'ba': mx.nd.zeros((n_labels, )),
    }

    from utils.timedate import timing
    import logging
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)

    opt_params = {'learning_rate': 0.01,
                  'gamma1': 0.5,
                  'gamma2': 0.8,
                  # decay=0.5,  # decay, gamma1
                  # momentum=0.8,  # momentum, gamma2
                  }
    # optimizer = mx.optimizer.RMSProp(**opt_params)
    eval_metric = mx.metric.create(loss_func)
    n_epochs = 20

    t0 = timing()
    if False:
        model = mx.model.FeedForward(
            # ctx=ctx,
            symbol=sym,
            num_epoch=n_epochs,
            optimizer='RMSProp',
            # optimizer_params=opt_params,
            # eval_metric=eval_metric,
            arg_params=arg_params,
            **opt_params
        )
        model.fit(X=data_train,
                  eval_metric=eval_metric,
                  batch_end_callback=mx.callback.Speedometer(batch_size, 20),
                  )
    else:
        module = mx.mod.Module(sym,
                               data_names=('data',),
                               label_names=('label',),
                               )
        module.bind(data_shapes=data_train.provide_data,
                    label_shapes=data_train.provide_label,
                    for_training=True,  # default
                    )
        module.init_params(arg_params=arg_params)

        if False:

            module.fit(data_train,
                       optimizer='RMSProp',  # mx.optimizer.RMSProp,
                       optimizer_params=opt_params,
                       num_epoch=n_epochs,
                       eval_metric=eval_metric,
                       )
        else:
            module.init_optimizer(kvstore='local',
                                  optimizer='RMSProp',
                                  optimizer_params=opt_params)

            for epoch in xrange(n_epochs):
                for idx, batch in enumerate(data_train):
                    module.forward(batch, is_train=True)
                    module.backward()
                    module.update()
                    module.update_metric(eval_metric=eval_metric,
                                         labels=batch.label)

                res = module.score(data_train, eval_metric)
                print res[0]

    timing(t0, 'mxnet', 's')
