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


def cnn():

    ep = 0.000001
    lamb = 0.0001
    p = 0.2
    p2 = 0.5
    learn_rate = 0.01
    mom = 0.001
    width, height = 28,28
    cnn = Sequential()

    # Convolutional layers, each ReLu activated with a dropout layer
    cnn.add(Convolution2D(filters=32,kernel_size=(3,3), padding = 'same',strides = (1,1), input_shape = (width, height,1),kernel_regularizer = l2(lamb),kernel_constraint=non_neg()))#, kernel_initializer=RandomUniform(minval=0, maxval=0.15, seed=None)))#,kernel_regularizer = l2(lamb)))
    cnn.add(Activation('relu'))
    # cnn.add(Dropout(p))
    cnn.add(Convolution2D(filters=64,kernel_size=(5,5), padding = 'same',strides = (1,1),kernel_regularizer = l2(lamb),kernel_constraint=non_neg()))#, kernel_initializer=RandomUniform(minval=0, maxval=0.15, seed=None)))#,kernel_regularizer = l2(lamb)))
    cnn.add(Activation('relu'))
    # cnn.add(Dropout(p))

    # Post-convolutional pooling layer, reducing the feature space by 75%.
    cnn.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))

    # Fully-connected layer feeding the output logits.
    # High dropout-rate layer after the fully-connected layer.
    cnn.add(Flatten())
    cnn.add(Dense(512,kernel_regularizer = l2(lamb),kernel_constraint=non_neg()))#,kernel_constraint=non_neg(), kernel_initializer=RandomUniform(minval=0, maxval=0.15, seed=None)))
    cnn.add(Activation('relu'))
    cnn.add(Dense(1024,kernel_regularizer = l2(lamb),kernel_constraint=non_neg()))#,kernel_constraint=non_neg(), kernel_initializer=RandomUniform(minval=0, maxval=0.15, seed=None)))
    cnn.add(Activation('relu'))
    # cnn.add(Dropout(p2))
    cnn.add(Dense(10,kernel_constraint=non_neg()))#,kernel_constraint=non_neg(), kernel_initializer=RandomUniform(minval=0, maxval=0.15, seed=None)))
    cnn.add(Activation('softmax'))

    A = Adadelta(lr=learn_rate, rho=0.95, epsilon=ep)
    # A = SGD(lr=learn_rate, momentum=mom, decay=0.0)
    cnn.compile(loss='kullback_leibler_divergence',optimizer=A,metrics=['accuracy'])

    cnn.summary()
    return cnn

def ffn():

    ep = 0.000001
    lamb = 0.00001
    p = 0.2
    p2 = 0.5
    learn_rate = 1
    mom = 0.001
    width, height = 28,28
    ffn = Sequential()

    ffn.add(Flatten(input_shape = (width, height,1)))

    ffn.add(Dense(16,kernel_regularizer = l2(lamb)))#,kernel_constraint=non_neg()))#,kernel_constraint=non_neg(), kernel_initializer=RandomUniform(minval=0, maxval=0.15, seed=None)))
    ffn.add(Activation('relu'))

    ffn.add(Dense(10))#,kernel_constraint=non_neg()))#,kernel_constraint=non_neg(), kernel_initializer=RandomUniform(minval=0, maxval=0.15, seed=None)))
    ffn.add(Activation('softmax'))

    A = Adadelta(lr=learn_rate, rho=0.95, epsilon=ep)
    ffn.compile(loss='categorical_crossentropy',optimizer=A,metrics=['accuracy'])

    ffn.summary()
    return ffn

def dsf():

    ep = 0.000001
    lamb = 0.00001
    learn_rate = 1
    width, height = 28,28
    dsf = Sequential()

    dsf.add(Flatten(input_shape = (width, height,1)))

    dsf.add(Dense(512,kernel_regularizer = l2(lamb),kernel_constraint=non_neg()))#,kernel_constraint=non_neg(), kernel_initializer=RandomUniform(minval=0, maxval=0.15, seed=None)))
    dsf.add(Activation('relu'))

    dsf.add(Dense(10,kernel_constraint=non_neg()))#,kernel_constraint=non_neg(), kernel_initializer=RandomUniform(minval=0, maxval=0.15, seed=None)))
    dsf.add(Activation('softmax'))

    A = Adadelta(lr=learn_rate, rho=0.95, epsilon=ep)
    dsf.compile(loss='categorical_crossentropy',optimizer=A,metrics=['accuracy'])

    dsf.summary()
    return dsf
