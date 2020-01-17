"""
Generates test data.

Loads the original model and runs the first 32 instances of the testdata through it.
Grabs the output of the 2nd convolutional layer and feeds that into a new model. The
model consist of only convolutional layer that uses the same weights and configuration
as the 2nd convolutional layer in the orginial model excpet the padding is `same` 
instead of `valid`.

The output of that model gets saved to a file. 

"""



import os
import tempfile
import keras
from keras import backend as K
from keras import datasets
import numpy as np

def square(x):
    return K.square(x)


a = keras.layers.Activation(square)

a.__name__ = 'square'

K.set_image_data_format( 'channels_first' )

keras.utils.get_custom_objects().update( { 'square': a } )
print( 'loading model' )
model = keras.models.load_model( os.path.join( 'data', 'mainModel' ) )
model.summary()

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
print( x_test.shape )
# channels first
x_train = x_train.reshape( x_train.shape[ 0 ], 1, x_train.shape[ 1 ], x_train.shape[ 2 ] )
x_test = x_test.reshape( x_test.shape[ 0 ], 1, x_test.shape[ 1 ], x_test.shape[ 2 ] )
print( x_test.shape )

inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers]          # all layer outputs
functor = K.function([inp, K.learning_phase()], outputs )   # evaluation function

# Testing
test = x_test[:32].astype(int)
layer_outs = functor([test, 0]) 


# gather data for new model
x = layer_outs[2]

