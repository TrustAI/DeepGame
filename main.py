from __future__ import print_function
from neural_network import *
from data_set import *

dataSetName = 'mnist'
#dataSetName = 'cifar10'


NN = NeuralNetwork(dataSetName)
# NN.train_network()
# NN.save_network()
NN.load_network()
print('Dataset is', NN.data_set)
NN.model.summary()

dataset = DataSet(dataSetName,'testing')
image = dataset.getInput(1)
label = NN.predict(image)
print(label)

