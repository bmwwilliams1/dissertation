from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.regularizers import l2 #, activity_l2
from keras.initializers import RandomUniform
import numpy
import csv
import scipy.misc
import scipy
import h5py
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.constraints import non_neg







def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()



(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], height, width, depth).astype('float32')
X_test = X_test.reshape(X_test.shape[0], height, width, depth).astype('float32')
