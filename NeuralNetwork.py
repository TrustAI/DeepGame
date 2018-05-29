"""
Construct a NeuralNetwork class to include operations
related to various datasets and corresponding models.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""

import keras
from keras.datasets import mnist, cifar10
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

from basics import assure_path_exists


# Define a Neural Network class.
class NeuralNetwork:
    # Specify which dataset at initialisation.
    def __init__(self, data_set):
        self.data_set = data_set
        self.model = Sequential()
        assure_path_exists("%s_pic/" % self.data_set)

    def predict(self, image):
        import numpy as np
        image = np.expand_dims(image, axis=0)
        predict_value = self.model.predict(image)
        new_class = np.argmax(np.ravel(predict_value))
        confident = np.amax(np.ravel(predict_value))
        return new_class, confident

    # To train a neural network.
    def train_network(self):
        # Train an mnist model.
        if self.data_set == 'mnist':
            batch_size = 128
            num_classes = 10
            epochs = 50
            img_rows, img_cols = 28, 28

            (x_train, y_train), (x_test, y_test) = mnist.load_data()

            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255

            y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
            model.add(Conv2D(32, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(200, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(200, activation='relu'))
            model.add(Dense(num_classes, activation='softmax'))

            model.compile(loss='categorical_crossentropy',
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])

            model.fit(x_train, y_train,
                      batch_size=batch_size,
                      epochs=epochs,
                      verbose=1,
                      validation_data=(x_test, y_test))

            score = model.evaluate(x_test, y_test, verbose=0)
            print("Test loss:", score[0])
            print("Test accuracy:", score[1])

            self.model = model

        # Train a cifar10 model.
        elif self.data_set == 'cifar10':
            batch_size = 128
            num_classes = 10
            epochs = 50
            img_rows, img_cols, img_chls = 32, 32, 3
            data_augmentation = True

            (x_train, y_train), (x_test, y_test) = cifar10.load_data()

            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_chls)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_chls)
            input_shape = (img_rows, img_cols, img_chls)

            x_train = x_train.astype('float32')
            x_test = x_test.astype('float32')
            x_train /= 255
            x_test /= 255

            y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
            y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

            model = Sequential()
            model.add(Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Flatten())
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(256, activation='relu'))
            model.add(Dense(num_classes, activation='softmax'))

            opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
            model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])

            if not data_augmentation:
                print("Not using data augmentation.")
                model.fit(x_train, y_train,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=(x_test, y_test),
                          shuffle=True)
            else:
                print("Using real-time data augmentation.")
                datagen = ImageDataGenerator(
                    featurewise_center=False,
                    samplewise_center=False,
                    featurewise_std_normalization=False,
                    samplewise_std_normalization=False,
                    zca_whitening=False,
                    rotation_range=0,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    horizontal_flip=True,
                    vertical_flip=False)

                datagen.fit(x_train)
                model.fit_generator(datagen.flow(x_train, y_train,
                                                 batch_size=batch_size),
                                    epochs=epochs,
                                    validation_data=(x_test, y_test),
                                    workers=4)

            scores = model.evaluate(x_test, y_test, verbose=0)
            print("Test loss:", scores[0])
            print("Test accuracy:", scores[1])

            self.model = model

        else:
            print("Unsupported dataset %s. Try 'mnist' or 'cifar10'." % self.data_set)

    # To save the neural network to disk.
    def save_network(self):
        if self.data_set == 'mnist':
            self.model.save('models/mnist.h5')
            print("Neural network saved to disk.")

        elif self.data_set == 'cifar10':
            self.model.save('models/cifar10.h5')
            print("Neural network saved to disk.")

        else:
            print("save_network: Unsupported dataset.")

    # To load a neural network from disk.
    def load_network(self):
        if self.data_set == 'mnist':
            self.model = load_model('models/mnist.h5')
            print("Neural network loaded from disk.")

        elif self.data_set == 'cifar10':
            self.model = load_model('models/cifar10.h5')
            print("Neural network loaded from disk.")

        else:
            print("load_network: Unsupported dataset.")

    def save_input(self, image, filename):

        import cv2
        import copy

        image_cv = copy.deepcopy(image)

        cv2.imwrite(filename, image_cv * 255.0, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    def get_label(self, index):
        if self.data_set == 'mnist':
            labels = ['0', '1', '2', '3', '4', '5', '6',
                      '7', '8', '9']
        elif self.data_set == 'cifar10':
            labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog',
                      'horse', 'ship', 'truck']
        else:
            print("LABELS: Unsupported dataset.")
        return labels[index]
