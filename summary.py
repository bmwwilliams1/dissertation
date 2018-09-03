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
import random as rd
import _pickle as pc
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import mnist
from keras.datasets import cifar10
from keras.datasets import reuters
from keras.constraints import non_neg
from keras.preprocessing.text import Tokenizer
import models
from keras.models import load_model
from keras.models import Model
from keras import layers
from keras import backend as back

# Load the models
load_name = 'Model.01-0.9699_train9910_32.hdf5'
model = load_model(load_name)
i=1
for layer in model.layers:
    if (layer.get_config()["name"].startswith("dense")):
        weights = layer.get_weights() # list of numpy arrays
        # print('weights dimensions: ',len(weights[0]),'x',len(weights[0][0]))
        print(layer.get_config())
        # np.savetxt("weights_FFNDEEP_%s.csv"%str(i),weights[0], delimiter=",")
        i=i+1


width = height = 28
classes = 10
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# X_test_trial = X_test[1]
# y_test_trial = y_test[1]


layer_name = 'dense_1'
int = Model(inputs = model.input, outputs = model.get_layer(layer_name).output)

results = np.zeros((10000,32),dtype = float)
for j in range(0,len(X_test)):
    intermediate_output = int.predict(np.reshape(X_test[j],(1,28,28,1)))
    # print(intermediate_output)
    results[j,:] = intermediate_output

np.savetxt("outputs_dsf.csv",results,delimiter = ",")
print(intermediate_output)

#
ffn = np.genfromtxt('outputs_ffn.csv',delimiter = ',')
# dsf = np.genfromtxt('outputs_dsf.csv',delimiter = ',')
ffn_weights = np.genfromtxt('weights_FFN_2.csv',delimiter = ',')
# dsf_weights = np.genfromtxt('weights_DSF_2.csv',delimiter = ',')


# dsf_after = np.dot(dsf,dsf_weights)
ffn_after = np.dot(ffn,ffn_weights)

# np.savetxt('dsf_after.csv',dsf_after,delimiter=',')
np.savetxt('ffn_after.csv',ffn_after,delimiter=',')
