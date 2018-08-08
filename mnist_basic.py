
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
import models

# import dataprocessing

# ~~~~~~~~~~~~~~~~~~~~~~~ MAIN FILE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def main():
    depth = 1
    epochs = 10
    classes = 10
    width = height = 28
    batch_size = 32
    val_split = 0.2

    # TRAIN, TEST or SAVE
    mode = 'TRAIN'
    load_name = 'Model.01-0.9670_test9745.hdf5'

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ~~~~~~~~~~~~~~~~~~~~~~~~ DEAL WITH THE DATA ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if mode == 'TRAIN':

        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = X_train.reshape(X_train.shape[0], height, width, depth).astype('float32')
        X_test = X_test.reshape(X_test.shape[0], height, width, depth).astype('float32')
        val_row = int(X_train.shape[0] * 0.8)

        # Rescale the inputs to [0,1]
        X_test /=255
        X_train /=255

        print(X_train.shape)
        print(X_test.shape)

        y_train = np_utils.to_categorical(y_train, classes)
        y_test = np_utils.to_categorical(y_test, classes)

        # Selecting the preprocessing if activated
        generate = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            rotation_range=0.0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images


        our_model = models.dsf()
        files='Model.{epoch:02d}-{val_acc:.4f}.hdf5'
        ckpt = keras.callbacks.ModelCheckpoint(files, monitor = 'val_loss',verbose=1, save_best_only=True, mode='auto')
        our_model.fit_generator(generate.flow(X_train, y_train), steps_per_epoch = X_train.shape[0], epochs=epochs,
                                validation_data = (X_test,y_test), callbacks = [ckpt])

    # test run on the test classes
    if mode == 'TEST':
        model = keras.models.load_model(load_name)
        incorrect_classes = np.nonzero(model.predict(X_test, batch_size=None, verbose=0, steps=None) != y_test)
        print(incorrect_classes)

    if mode == 'SAVE':
        model = load_model(load_name)
        weights = []
        for layer in model.layers:
            weights.append.layer.get_weights() # list of numpy arrays

        np.savetxt("weights.csv",weights, delimiter=",")


if __name__ == "__main__":
    main()
