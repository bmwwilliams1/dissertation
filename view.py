from __future__ import print_function
import keras
import h5py
import numpy as np
from keras.models import load_model


model = load_model('Model.01-0.9704.hdf5')
keras.utils.print_summary(model, line_length=None, positions=None, print_fn=None)
keras.utils.plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')

for layer in model.layers:
    weights = layer.get_weights() # list of numpy arrays
    print(weights)
