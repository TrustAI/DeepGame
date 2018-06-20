"""
Construct a NeuralNetwork class to include operations
related to various datasets and corresponding models.

Author: Min Wu
Email: min.wu@cs.ox.ac.uk
"""

import keras
from keras.datasets import mnist, cifar10
from skimage import io, color, exposure, transform
import pandas as pd
import numpy as np
import h5py
import os
import glob


# Define a Neural Network class.
class DataSet:
    # Specify which dataset at initialisation.
    def __init__(self, data_set, trainOrTest):
        self.data_set = data_set

        # for a mnist model.
        if self.data_set == 'mnist':
            num_classes = 10
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            img_rows, img_cols, img_chls = 28, 28, 1
            if trainOrTest == "training":
                x = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_chls)
                y = keras.utils.np_utils.to_categorical(y_train, num_classes)
            else:
                x = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_chls)
                y = keras.utils.np_utils.to_categorical(y_test, num_classes)

            x = x.astype('float32')
            x /= 255

        # for a cifar10 model.
        elif self.data_set == 'cifar10':
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

        # for a gtsrb model.
        elif self.data_set == 'gtsrb':
            num_classes = 43
            img_rows, img_cols, img_chls = 48, 48, 3
            if trainOrTest == "training":
                directory = 'models/GTSRB/Final_Training/'
                try:
                    with h5py.File(directory + 'gtsrb_training.h5') as hf:
                        x_train, y_train = hf['imgs'][:], hf['labels'][:]
                    x = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_chls)
                    y = keras.utils.np_utils.to_categorical(y_train, num_classes)

                except (IOError, OSError, KeyError):
                    imgs = []
                    labels = []
                    all_img_paths = glob.glob(os.path.join(directory + 'Images/', '*/*.ppm'))
                    np.random.shuffle(all_img_paths)
                    for img_path in all_img_paths:
                        try:
                            img = self.preprocess_img(io.imread(img_path), img_rows, img_cols)
                            label = self.get_class(img_path)
                            imgs.append(img)
                            labels.append(label)

                            if len(imgs) % 1000 == 0: print("Processed {}/{}".format(len(imgs), len(all_img_paths)))
                        except (IOError, OSError):
                            print('missed', img_path)
                            pass

                    x_train = np.array(imgs, dtype='float32')
                    y_train = np.array(labels, dtype='uint8')

                    with h5py.File(directory + 'gtsrb_training.h5', 'w') as hf:
                        hf.create_dataset('imgs', data=x_train)
                        hf.create_dataset('labels', data=y_train)
                    x = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_chls)
                    y = keras.utils.np_utils.to_categorical(y_train, num_classes)

            else:
                directory = 'models/GTSRB/Final_Test/'
                try:
                    with h5py.File(directory + 'gtsrb_test.h5') as hf:
                        x_test, y_test = hf['imgs'][:], hf['labels'][:]
                    x = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_chls)
                    y = keras.utils.np_utils.to_categorical(y_test, num_classes)

                except (IOError, OSError, KeyError):
                    test = pd.read_csv(directory + 'GT-final_test.csv', sep=';')
                    x_test = []
                    y_test = []
                    for file_name, class_id in zip(list(test['Filename']), list(test['ClassId'])):
                        img_path = os.path.join(directory + 'Images/', file_name)
                        x_test.append(self.preprocess_img(io.imread(img_path), img_rows, img_cols))
                        y_test.append(class_id)

                    x_test = np.array(x_test, dtype='float32')
                    y_test = np.array(y_test, dtype='uint8')

                    with h5py.File(directory + 'gtsrb_test.h5', 'w') as hf:
                        hf.create_dataset('imgs', data=x_test)
                        hf.create_dataset('labels', data=y_test)
                    x = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_chls)
                    y = keras.utils.np_utils.to_categorical(y_test, num_classes)

        else:
            print("Unsupported dataset %s. Try 'mnist' or 'cifar10'." % data_set)
            exit()

        self.x = x
        self.y = y

    # get dataset 
    def get_dataset(self):
        return self.x, self.y

    def get_input(self, index):
        return self.x[index]

    def preprocess_img(self, img, img_rows, img_cols):
        # Histogram normalization in y
        hsv = color.rgb2hsv(img)
        hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
        img = color.hsv2rgb(hsv)

        # central scrop
        min_side = min(img.shape[:-1])
        centre = img.shape[0] // 2, img.shape[1] // 2
        img = img[centre[0] - min_side // 2:centre[0] + min_side // 2,
              centre[1] - min_side // 2:centre[1] + min_side // 2, :]

        # rescale to standard size
        img = transform.resize(img, (img_rows, img_cols))

        # roll color axis to axis 0
        # img = np.rollaxis(img, -1)

        return img

    def get_class(self, img_path):
        return int(img_path.split('/')[-2])
