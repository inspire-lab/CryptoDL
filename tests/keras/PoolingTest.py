import keras
from keras.models import Sequential
from keras.layers import Conv2D, AveragePooling2D
from keras import backend as K
import numpy as np
from prompt_toolkit import input
import sys


def square(x):
    return K.square(x)
keras.utils.get_custom_objects().update( { 'square': keras.layers.Activation(square) } )

K.set_image_data_format( 'channels_first' )


############## Basic Average Pooling ########################
'''
#t_input = np.arange(1, 37).reshape((1,1,6,6))
t_input = np.ones((1,1,9,9))

model = Sequential()
model.add( Conv2D( 1, (3, 3), strides=(1,1), activation='linear', padding='same', input_shape=t_input.shape[1:] ) )
model.add(AveragePooling2D(pool_size=(2,2), strides = None, padding='valid', data_format=None))
#
model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
model.summary()

new_weights = np.array([np.ones((3, 3, 1, 1)), np.array([0])])
#new_weights = np.array([np.arange(1, 5).reshape((2,2,1,1)), np.array([0])])

model.layers[0].set_weights(new_weights)

print("Weights")
print(model.layers[0].get_weights()[0])

print("Input:")
print(input)
print("Output:")
print(model.predict(t_input))'''

t_input = np.arange(1, 37).reshape((1,1,6,6))
#t_input = np.ones((1,1,10,10))

model = Sequential()
model.add(AveragePooling2D(pool_size=(2,2), strides = (2,2), padding='valid', data_format=None, input_shape=t_input.shape[1:]))
model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
model.summary()

print("Input:")
print(t_input)
print("Output:")
print(model.predict(t_input))













