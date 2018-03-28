
from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.regularizers import l2 #, activity_l2
import numpy
import csv
import scipy.misc
import scipy

import numpy as np
import h5py
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
# import dataprocessing



def cnn():

    ep = 0.000001
    lamb = 0.00001
    p = 0.2
    p2 = 0.5
    learn_rate = 0.01
    width, height = 28,28
    cnn = Sequential()


    cnn.add(Convolution2D(filters=32,kernel_size=(3,3),kernel_regularizer = l2(lamb), padding = 'same',strides = (1,1), input_shape = (width, height,1)))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(p))
    # cnn.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    cnn.add(Convolution2D(filters=64,kernel_size=(3,3),kernel_regularizer = l2(lamb), padding = 'same',strides = (1,1)))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(p))
    # cnn.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    # cnn.add(Convolution2D(filters=512,kernel_size=(3,3), padding = 'same',strides = (1,1)))
    # cnn.add(Activation('relu'))
    # cnn.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    # cnn.add(Convolution2D(filters=128,kernel_size=(3,3),kernel_regularizer = l2(lamb), padding = 'same',strides = (1,1)))
    # cnn.add(Activation('relu'))
    # cnn.add(Dropout(p))
    cnn.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))

    cnn.add(Flatten())
    cnn.add(Dense(256))
    cnn.add(Activation('relu'))
    cnn.add(Dropout(p2))
    cnn.add(Dense(10))
    cnn.add(Activation('softmax'))




    A = Adadelta(lr=learn_rate, rho=0.95, epsilon=ep)
    cnn.compile(loss='categorical_crossentropy',optimizer=A,metrics=['accuracy'])

    cnn.summary()
    return cnn

# ~~~~~~~~~~~~~~~~~~~~~~~ MAIN FILE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

depth = 1
epochs = 20
classes = 10
width = height = 28
batch_size = 32
val_split = 0.2


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~ DEAL WITH THE DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], height, width, depth).astype('float32')
X_test = X_test.reshape(X_test.shape[0], height, width, depth).astype('float32')
val_row = int(X_train.shape[0] * 0.8)
# print(val_row)
X_val = X_train[val_row:]
y_val = y_train[val_row:]
X_train = X_train[:val_row]

y_train = y_train[:val_row]

# print("x_train shape: ",X_train.shape)
# print("x_val shape: ",X_val.shape)


y_train = np_utils.to_categorical(y_train, classes)
y_val = np_utils.to_categorical(y_train, classes)
y_test = np_utils.to_categorical(y_test, classes)

generate = ImageDataGenerator(
    featurewise_center=True,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=True,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False)  # randomly flip images

our_model = cnn()
files='Model.{epoch:02d}-{val_acc:.4f}.hdf5'
ckpt = keras.callbacks.ModelCheckpoint(files, monitor = 'val_loss',verbose=1, save_best_only=True, mode='auto')
# print(our_model.summary())
# generate.fit(X_train)
our_model.fit_generator(generate.flow(X_train, y_train), steps_per_epoch = X_train.shape[0], epochs=epochs,
                        validation_data=(X_test,y_test), callbacks = [ckpt])
