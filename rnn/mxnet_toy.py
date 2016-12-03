#!/usr/bin/env python

import mxnet as mx

a = mx.nd.zeros((100, 50))
b = mx.nd.ones((100, 50))
c = a + b
c += 1
print(c)

net = mx.symbol.Variable('data')
net = mx.symbol.FullyConnected(data=net, num_hidden=3)
net = mx.symbol.SoftmaxOutput(data=net)

mod = mx.mod.Module(net)
mod.forward(data=c)
mod.backward()
