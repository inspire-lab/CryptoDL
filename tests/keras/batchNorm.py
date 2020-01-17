# 
# Used to generate baselines for other testcases
#

import keras
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization
from keras import backend as K
import numpy as np


def square(x):
    return K.square(x)
keras.utils.get_custom_objects().update( { 'square': keras.layers.Activation(square) } )


K.set_image_data_format( 'channels_first' )

############### Flatten Test 1 ##################################
print('\n\n\nConv Test 1 same padding')
 
input = np.array( [ [ [ [ 1,1,1,1 ], [ 1,1,1,1 ],[ 1,1,1,1 ],[ 1,1,1,1 ] ] ],
                    [ [ [ 2,2,2,2 ], [ 2,2,2,2 ],[ 2,2,2,2 ],[ 2,2,2,2 ] ] ],
                    [ [ [ 3,3,3,3 ], [ 3,3,3,3 ],[ 3,3,3,3 ],[ 3,3,3,3 ] ] ]
                    ] )
print(input.shape)
  
model = Sequential()
model.add( Conv2D( 1, (3,3), strides=(1,1), activation='square', padding='same', input_shape=input.shape[1:] ) )
# model.add( BatchNormalization() )
model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
model.summary()
  
# filter shape (3, 3, 1, 1) and shape (1) bias
weights = [np.array(  [[[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]]] ), np.array([1])]
model.layers[0].set_weights(weights)

# print( model.layers[1].get_weights() )
  
print(model.predict(input[ : 2 ] ) )


