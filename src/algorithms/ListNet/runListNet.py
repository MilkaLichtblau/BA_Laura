# -*- coding: utf-8 -*-
"""
Created on Sun May 27 19:36:19 2018

@author: Laura
"""

import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


x_data = np.array([5], dtype=np.float32)
x = Variable(x_data)

z = 2*x
y = x**2 - z + 1
y.backward(retain_grad=True)

print(z.grad)