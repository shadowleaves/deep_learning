""" Simple multi-layer perception neural network using Minpy """
# import minpy
import minpy.numpy as np
from minpy.nn import layers
from minpy.nn.model import ModelBase
from minpy.nn.solver import Solver
from minpy.nn.io import NDArrayIter
from examples.utils.data_utils import get_CIFAR10_data
from minpy.context import set_context, gpu

# set_context(gpu(0))  # set the global context as gpu(0)


batch_size = 128
input_size = (3, 32, 32)

#                     RGB  width height
flattened_input_size = 3 * 32 * 32
hidden_size = 512
num_classes = 10


class TwoLayerNet(ModelBase):
    def __init__(self):
        super(TwoLayerNet, self).__init__()
        # Define model parameters.
        self.add_param(name='w1', shape=(flattened_input_size, hidden_size))
        self.add_param(name='b1', shape=(hidden_size,))
        self.add_param(name='w2', shape=(hidden_size, num_classes))
        self.add_param(name='b2', shape=(num_classes,))

    def forward(self, X, mode):
        # Flatten the input data to matrix.

        # three channel RGB picture - thus the 3* below...
        X = np.reshape(X, (batch_size, 3 * 32 * 32))
        # First affine layer (fully-connected layer).
        y1 = layers.affine(X, self.params['w1'], self.params['b1'])
        # ReLU activation.
        y2 = layers.relu(y1)
        # Second affine layer.
        y3 = layers.affine(y2, self.params['w2'], self.params['b2'])

        return y3

    def loss(self, predict, y):
        # Compute softmax loss between the output and the label.
        # this function must be convex for gradient calculation
        return layers.softmax_loss(predict, y)


def main():

    # data dir
    import os
    data_dir = os.path.expandvars('$HOME/data/minpy/cifar-10-batches-py')

    # Create model.
    model = TwoLayerNet()
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
                        'learning_rate': 1e-4,
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
