# 
# Used to generate baselines for other testcases
#

import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras import backend as K
import numpy as np


def square(x):
    return K.square(x)
keras.utils.get_custom_objects().update( { 'square': keras.layers.Activation(square) } )


K.set_image_data_format( 'channels_first' )

############### Flatten Test 1 ##################################
print('\n\n\nFlatten Test')

input = np.array( [ [ [ [ 1,2,3,4 ], [ 5,6,7,8 ] , [ 9,10,11,12 ],[ 13,14,15,16 ] ] ] ] )

print(input)

model = Sequential()
model.add( Flatten( input_shape=input.shape[1:] ) )
model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
model.summary()

print(model.predict(input))


############### Flatten Test 2 ##################################
print('\n\n\nFlatten Test 2')

input = np.array( [ [ [ [ 1,2,3,4 ], [ 5,6,7,8 ] ] , [ [ 9,10,11,12 ],[ 13,14,15,16 ] ] ] ] )



print(input)
print(input.shape)

model = Sequential()
model.add( Flatten( input_shape=input.shape[1:] ) )
model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
model.summary()

print( model.layers[0].get_config() )

print(model.predict(input))

############### Dense Test 1 ##################################
print('\n\n\nDense Test 1')

input = np.ones( (1, 6) )

print(input)

model = Sequential()
model.add( Dense( 3, activation='square', input_shape=input.shape[1:] ) )
model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
model.summary()

# print( model.layers[0].get_weights() )
# print( model.layers[0].get_weights()[0].shape )

# weights shape (6, 3) and shape (3) bias
weights = [np.ones( (6,3) ), np.ones( 3 )]
model.layers[0].set_weights(weights)

print(model.predict(input))
############### Dense Test 2 ##################################
print('\n\n\nDense Test 2')

input = np.ones( (2, 6) )

print(input)

model = Sequential()
model.add( Dense( 3, activation='square', input_shape=input.shape[1:] ) )
model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
model.summary()

# print( model.layers[0].get_weights() )
# print( model.layers[0].get_weights()[0].shape )

# weights shape (6, 3) and shape (3) bias
weights = [np.ones( (6,3) ), np.ones( 3 )]
model.layers[0].set_weights(weights)

print(model.predict(input))


############### Dense Test 3 ##################################
print('\n\n\nDense Test 3')

input = np.ones( (2, 6000) ) * 10000

print(input)

model = Sequential()
model.add( Dense( 10, activation='square', input_shape=input.shape[1:] ) )
model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
model.summary()

# print( model.layers[0].get_weights() )
# print( model.layers[0].get_weights()[0].shape )

# weights shape (6000, 3000) and shape (3000) bias
weights = [np.ones( (6000,10) )*10000, np.ones( 10 )*10000]
model.layers[0].set_weights(weights)

print(model.predict(input).shape)
print( int( model.predict( input )[0][0] ) )

############### Dense Test 4 ##################################
print('\n\n\nDense Test 4')

input = np.arange( 12 ).reshape( (2,6) )

print(input)

model = Sequential()
model.add( Dense( 3, activation='square', input_shape=input.shape[1:], data_format='channels_last' ) )
model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
model.summary()

# print( model.layers[0].get_weights() )
# print( model.layers[0].get_weights()[0].shape )

# weights shape (6, 3) and shape (3) bias
weights = [np.arange( 18 ).reshape((6,3)), np.zeros( 3 )]
model.layers[0].set_weights(weights)

print(model.predict(input))

