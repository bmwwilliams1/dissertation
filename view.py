from __future__ import print_function
import keras
import h5py
import numpy as np
from keras.models import load_model
from keras.models import Model


model = load_model('Model.03-0.9718.hdf5')
# keras.utils.print_summary(model, line_length=None, positions=None, print_fn=None)
# keras.utils.plot_model(model, to_file='model.png', show_shapes=False, show_layer_names=True, rankdir='TB')

# weights = []
# for layer in model.layers:
#     weights.append.layer.get_weights() # list of numpy arrays
weights = model.layers[0].get_weights()

np.savetxt("weights_512.csv",weights, delimiter=",")
