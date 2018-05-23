from __future__ import print_function
from neural_network import *


MNIST = NeuralNetwork('mnist')
# MNIST.train_network()
# MNIST.save_network()
MNIST.load_network()
print('Dataset is', MNIST.data_set)
MNIST.model.summary()


CIFAR10 = NeuralNetwork('cifar10')
# CIFAR10.train_network()
# CIFAR10.save_network()
CIFAR10.load_network()
print('Dataset is', CIFAR10.data_set)
CIFAR10.model.summary()


