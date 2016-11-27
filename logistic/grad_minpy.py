# do not use the following line - incompatible with GPU/CUDA
# !/usr/bin/env python

import minpy.numpy as np
import minpy.numpy.random as random
from minpy.core import grad_and_loss
from examples.utils.data_utils import gaussian_cluster_generator as make_data
from minpy.context import set_context, gpu

# Please uncomment following if you have GPU-enabled MXNet installed.
# This single line of code will run MXNet operations on GPU 0.
set_context(gpu(0))  # set the global context as gpu(0)
# set_context(cpu())  # set the global context as cpu

# Predict the class using multinomial logistic regression (softmax regression).


def predict(w, x):
    # actually soft-max function in Hinton's book
    # yi = exp(zi) / sum(exp(zj))  for i, j in (1, 2, 3 ...)
    a = np.exp(np.dot(x, w))
    a_sum = np.sum(a, axis=1, keepdims=True)
    prob = a / a_sum
    return prob


def train_loss(w, x, label):
    prob = predict(w, x)
    num_samples, num_features = x.shape

    # standard log likelihood loss function with binary label
    # this is the CE (cross-entropy) in Hinton's course
    loss = -np.sum(label * np.log(prob)) / num_samples
    return loss


def train(w, x, label, loops, lr=0.1):
    '''Using gradient descent to fit the correct classes.'''

    # abstract gradient function
    grad_function = grad_and_loss(train_loss)

    # main loop
    for i in range(loops):
        dw, loss = grad_function(w, x, label)
        if i % 10 == 0:
            print('Iter {}, training loss {}'.format(i, loss))
        # gradient descent
        w -= lr * dw


def main():
    """Use Minpy's auto-grad to derive a gradient function off loss"""

    # Initialize training data.
    num_samples = 10000
    num_features = 500
    num_classes = 5
    data, label = make_data(num_samples, num_features, num_classes)

    # Initialize training weight and train
    weight = random.randn(num_features, num_classes)
    # weight = np.ones((num_features, num_classes)) / 5.0
    train(weight, data, label, 100)


if __name__ == '__main__':
    main()
