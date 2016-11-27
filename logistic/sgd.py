# do not use the following line - incompatible with GPU/CUDA on Centos 7
# !/usr/bin/env python

import minpy.numpy as np
import minpy.numpy.random as random
from minpy.core import grad_and_loss
from examples.utils.data_utils import gaussian_cluster_generator as make_data
from minpy.context import set_context, gpu

from utils.timedate import timing
# Please uncomment following if you have GPU-enabled MXNet installed.
# This single line of code will run MXNet operations on GPU 0.
# set_context(gpu(0))  # set the global context as gpu(0)
# set_context(cpu())  # set the global context as cpu

# Predict the class using multinomial logistic regression (softmax regression).


def predict(w, x):
    # actually soft-max function in Hinton's book
    # yi = exp(zi) / sum(exp(zj))  for i, j in (1, 2, 3 ...)
    # zi = x . w
    a = np.exp(np.dot(x, w))
    a_sum = np.sum(a, axis=1, keepdims=True)
    prob = a / a_sum

    return prob


def train_loss(w, x, label):
    prob = predict(w, x)
    num_samples, num_features = x.shape

    # standard log likelihood loss function with binary label (see Ian's book)
    # this is the CE (cross-entropy) in Hinton's course
    # formula (8.9) page 281 of Ian's book (before the gradient jacobian)
    loss = -np.sum(label * np.log(prob))[0] / num_samples
    return loss


def manual_grad(w, x, t):
    y = predict(w, x)
    num_samples, num_features = x.shape
    loss = -np.sum(t * np.log(y))[0] / num_samples

    # grad for cross-entropy: dCE/dz = (y - t),  z = x dot w
    grad = (y - t) / y.shape[0]
    # chain rules, adding grad for input layer : dz/dw = x'
    grad = np.dot(x.T, grad)
    return grad, loss

def train(w, x, label, lr=0.01, minibatch_size=100, n_epochs=50,
          momentum=0.9, autograd=False):
    '''Using gradient descent to fit the correct classes.'''

    # abstract gradient function
    grad_function = grad_and_loss(train_loss)

    n_valid = 1000
    n_test = 1000

    # main loop
    n_sample, n_wgt = x.shape
    idx = range(n_sample)
    shuffled = np.random.permutation(idx)

    valid_set = shuffled[:n_valid].asnumpy()
    test_set = shuffled[n_valid:n_valid + n_test].asnumpy()
    training_set = shuffled[n_valid + n_test:].asnumpy()

    n_tr = training_set.shape[0]
    batches = int(n_tr / minibatch_size)
    # epoch_size = int(batches / n_epochs)
    valid_size = int(n_valid / n_epochs)
    j = 0

    # iteration
    dw = 0.

    for i in xrange(n_epochs):
        print 'epoch # %d:' % (i + 1)
        for k in xrange(batches):
            # training data for the minibatch
            start = k * minibatch_size
            picked = training_set[start:start + minibatch_size]
            # loss and gradient
            # t0 = timing()
            if autograd:
                grad, loss = grad_function(w, x[picked], label[picked])
            else:
                grad, loss = manual_grad(w, x[picked], label[picked])
            # timing(t0, 'grad+loss', 'ms')
            if float(loss) != float(loss):
                break

            # gradient descent with Sutskever / Nesterov momentum
            dw = momentum * dw + grad  # / minibatch_size
            w -= dw * lr

            if k % 10 == 0:
                print 'minibatch # {}, training loss = {}'.format(k, loss)

        picked = valid_set[j:j + valid_size]
        valid_loss = train_loss(w, x[picked], label[picked])
        print 'valid loss = {}'.format(valid_loss)
        j = j + valid_size

    # final test
    test_loss = train_loss(w, x[test_set], label[test_set])
    print 'final test loss = {}'.format(test_loss)


if __name__ == '__main__':
    """Use Minpy's auto-grad to derive a gradient function off loss"""

    # Initialize training data.
    num_samples = 10000
    num_features = 500
    num_classes = 5
    data, label = make_data(num_samples, num_features, num_classes)

    # Initialize training weight and train
    weight = random.randn(num_features, num_classes)
    # weight = np.ones((num_features, num_classes)) / 5.0
    train(weight, data, label, autograd=False)
