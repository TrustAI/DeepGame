"""
Construct a NeuralNetwork class to include operations
related to various datasets and corresponding models.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""


import keras
from keras.datasets import mnist, cifar10


# Define a Neural Network class.
class DataSet:
    # Specify which dataset at initialisation.
    def __init__(self, data_set, trainOrTest):
        self.data_set = data_set

        # for a mnist model.
        if self.data_set is 'mnist':
            num_classes = 10
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            img_rows, img_cols = 28, 28
            if trainOrTest == "training": 
                x = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
                y = keras.utils.np_utils.to_categorical(y_train, num_classes)
            else: 
                x = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
                y = keras.utils.np_utils.to_categorical(y_test, num_classes)

            x = x.astype('float32')
            x /= 255

        # for a cifar10 model.
        elif self.data_set is 'cifar10':
            num_classes = 10
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            img_rows, img_cols, img_chls = 32, 32, 3
            if trainOrTest == "training": 
                x = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_chls)
                y = keras.utils.np_utils.to_categorical(y_train, num_classes)
            else: 
                x = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_chls)
                y = keras.utils.np_utils.to_categorical(y_test, num_classes)
                
            x = x.astype('float32')
            x /= 255
        
        else:
            print("Unsupported dataset. Try 'mnist' or 'cifar10'.")
            exit()
            
        self.x = x
        self.y = y
            
    
    # get dataset 
    def getDataSet(self): 
        return (self.x, self.y)

    def getInput(self,index): 
        return self.x[index]
        
    