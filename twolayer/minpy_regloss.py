""" Simple multi-layer perception neural network using Minpy """
# import minpy
import minpy.numpy as np
from minpy.core import Function
from minpy.nn import layers
from minpy.nn.model import ModelBase
from minpy.nn.solver import Solver
from minpy.nn.io import NDArrayIter
from examples.utils.data_utils import get_CIFAR10_data
from minpy.context import set_context, gpu

# set_context(gpu(0))  # set the global context as gpu(0)
import mxnet as mx

batch_size = 128
input_size = (3, 32, 32)
flattened_input_size = 3 * 32 * 32
hidden_size = 512
num_classes = 10
weight_decay = 0.001


class ConvolutionNet(ModelBase):
    def __init__(self):
        super(ConvolutionNet, self).__init__()
        # Define symbols that using convolution
        # and max pooling to extract better features
        # from input image.
        net = mx.sym.Variable(name='X')
        net = mx.sym.Convolution(
            data=net, name='conv', kernel=(7, 7), num_filter=32)
        net = mx.sym.Activation(
            data=net, act_type='relu')
        net = mx.sym.Pooling(
            data=net, name='pool', pool_type='max', kernel=(2, 2),
            stride=(2, 2))
        net = mx.sym.Flatten(data=net)
        net = mx.sym.FullyConnected(
            data=net, name='fc1', num_hidden=hidden_size)
        net = mx.sym.Activation(
            data=net, act_type='relu')
        net = mx.sym.FullyConnected(
            data=net, name='fc2', num_hidden=num_classes)
        net = mx.sym.SoftmaxOutput(
            data=net, name='softmax', normalization='batch')
        # Create forward function and add parameters to this model.
        self.cnn = Function(
            net, input_shapes={'X': (batch_size,) + input_size,
                               'softmax_label': (batch_size,)},
            name='cnn')
        self.add_params(self.cnn.get_params())

    def forward_batch(self, batch, mode):
        out = self.cnn(X=batch.data[0],
                       softmax_label=batch.label[0],
                       **self.params)
        return out

    # def loss(self, predict, y):
    #     return layers.softmax_cross_entropy(predict, y)

    def loss(self, predict, y):
        # Add L2 regularization for all the weights.
        reg_loss = 0.0
        for name, weight in self.params.items():
            reg_loss += np.sum(weight ** 2)
        return layers.softmax_cross_entropy(predict, y) + \
            0.5 * weight_decay * reg_loss


def main():

    # data dir
    import os
    data_dir = os.path.expandvars('$HOME/data/minpy/cifar-10-batches-py')

    # Create model.
    model = ConvolutionNet()
    # Create data iterators for training and testing sets.
    data = get_CIFAR10_data(data_dir)
    train_dataiter = NDArrayIter(data=data['X_train'],
                                 label=data['y_train'],
                                 batch_size=batch_size,
                                 shuffle=True)
    test_dataiter = NDArrayIter(data=data['X_test'],
                                label=data['y_test'],
                                batch_size=batch_size,
                                shuffle=False)
    # Create solver.
    solver = Solver(model,
                    train_dataiter,
                    test_dataiter,
                    num_epochs=10,
                    init_rule='gaussian',
                    init_config={
                        'stdvar': 0.001
                    },
                    # automatically does the backpropagation
                    update_rule='sgd_momentum',
                    optim_config={
                        'learning_rate': 1e-3,
                        'momentum': 0.9
                    },
                    verbose=True,
                    print_every=20)
    # Initialize model parameters.
    solver.init()
    # Train!
    solver.train()


if __name__ == '__main__':
    main()