# 
# Used to generate baselines for other testcases
#

import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras import backend as K
import numpy as np
from prompt_toolkit import input
import sys


def square(x):
    return K.square(x)
keras.utils.get_custom_objects().update( { 'square': keras.layers.Activation(square) } )

K.set_image_data_format( 'channels_first' )

# ############### Conv Test 1 ##################################
# print('\n\n\nConv Test 1 same padding')
#  
# input = np.array( [ [ [ [ 1,1,1,1 ], [ 1,1,1,1 ],[ 1,1,1,1 ],[ 1,1,1,1 ] ] ] ] )
# print(input.shape)
#  
# model = Sequential()
# model.add( Conv2D( 1, (3,3), strides=(1,1), activation='square', padding='same', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
#  
# # filter shape (3, 3, 1, 1) and shape (1) bias
# weights = [np.array(  [[[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]]] ), np.array([1])]
# model.layers[0].set_weights(weights)
#  
# print(model.predict(input))
#  
# ############### Conv Test 2 ##################################
# print('\n\n\nConv Test 2 same padding')
#  
# input = np.array( [ [ [ [ 1,2,2,1 ], [ 1,2,2,1],[ 1,2,2,1 ],[ 1,2,2,1 ] ] ] ] )
#  
# model = Sequential()
# model.add( Conv2D( 1, (3,3), strides=(1,1), activation='square', padding='same', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
#  
#  
# # filter shape (3, 3, 1, 1) and shape (1) bias
# weights = [np.array(  [[[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]]] ), np.array([1])]
# model.layers[0].set_weights(weights)
# 
# 
# print(model.predict(input))
#  
# ############### Conv Test 3 ##################################
# print('\n\n\nConv Test 3 same padding')
#  
# input = np.array( [ [ [ [ 2,2,3,2 ], [ 2,1,2,3 ],[ 1,3,2,3 ],[ 2,2,1,2 ] ] ] ] )
#  
# model = Sequential()
# model.add( Conv2D( 1, (3,3), strides=(1,1), activation='square', padding='same', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
#  
# # filter shape (3, 3, 1, 1) and shape (1) bias
# weights = [np.array(  [[[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]]] ), np.array([1])]
# model.layers[0].set_weights(weights)
#  
# print(model.predict(input))
#  
# ############### Conv Test 4 ##################################
# print('\n\n\nConv Test 4 same padding')
#  
# input = np.array( [ [ [ [ 1,3,3,2 ], [ 2,1,3,2 ],[ 1,3,2,1 ],[ 3,2,3,1 ] ] ] ] )
#  
# model = Sequential()
# model.add( Conv2D( 1, (3,3), strides=(1,1), activation='square', padding='same', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
#  
# # filter shape (3, 3, 1, 1) and shape (1) bias
# weights = [np.array(  [[[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]]] ), np.array([1])]
# model.layers[0].set_weights(weights)
#  
# print(model.predict(input))
#  
# ############### Conv Test 5 ##################################
# print('\n\n\nConv Test 5 same padding')
# input = np.array( [ [ [ [ 3,2,3,2 ], [ 2,4,1,3 ],[ 2,2,1,3 ],[ 1,3,2,3 ] ] ] ] )
#  
# model = Sequential()
# model.add( Conv2D( 1, (3,3), strides=(1,1), activation='square', padding='same', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
#  
# # filter shape (3, 3, 1, 1) and shape (1) bias
# weights = [np.array(  [[[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]]] ), np.array([1])] 
# model.layers[0].set_weights(weights)
#  
# print(model.predict(input))
#  
# ############### Conv Test 6 ##################################
# print('\n\n\nConv Test 6 same padding')
# input = np.array( [ [ [ [ 1,2,3,4 ], [ 5,6,7,8 ],[ 9,10,11,12 ],[ 13,14,15,16 ] ] ] ] )
#  
# model = Sequential()
# model.add( Conv2D( 1, (3,3), strides=(1,1), activation='square', padding='same', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
#  
# # filter shape (3, 3, 1, 1) and shape (1) bias
# weights = [np.array(  [[[[1]],[[2]],[[3]]], [[[4]],[[5]],[[6]]], [[[7]],[[8]],[[9]]]] ), np.array([1])] 
# model.layers[0].set_weights(weights)
#  
# print(model.predict(input))
# 
# ############### Conv Test 7 ##################################
# print('\n\n\nConv Test 7 same padding')
# x=0.001
# input = np.ones( (1,1,4,4) ).astype(float)*x
# 
# model = Sequential()
# model.add( Conv2D( 1, (3,3), strides=(1,1), activation='square', padding='same', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
# 
# np.set_printoptions(precision=None)
# # filter shape (3, 3, 1, 1) and shape (1) bias
# weights = [np.ones( (3,3,1,1) ).astype(float)*x, np.array([1]).astype(float)] 
# model.layers[0].set_weights(weights)
# 
# print( model.layers[0].get_weights()[0] )
# print(model.predict(input).reshape(-1))
# 
# ############### Conv Test strides 1 same padding ##################################
# print('\n\n\nConv Test strides 1 same padding')
#  
# input = np.ones((1, 1, 16, 16) )
#  
# model = Sequential()
# model.add( Conv2D( 1, (3,3), strides=(2,2), activation='square', padding='same', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
#  
# # filter shape (3, 3, 1, 1) and shape (1) bias
# weights = [np.array(  [[[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]]] ), np.array([1])] 
# model.layers[0].set_weights(weights)
#  
# print(model.predict(input))
#  
# ############### Conv Test multiple channels non unit stride same padding 1  ##################################
# print('\n\n\nConv Test multiple channels non unit stride same padding 1')
#  
# input = np.ones( (1, 3, 16, 16) )
#  
# model = Sequential()
# model.add( Conv2D( 1, (3,3), strides=(2,2), activation='square', padding='same', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
#  
#  
# # filter shape (3, 3, 3, 1) and shape (1) bias
# weights = [ np.ones( (3, 3, 3, 1) ) , np.array([1])]
# model.layers[0].set_weights(weights)
#  
# print(model.predict(input))
#  
# ############### Conv Test multiple filters multiple channels non unit stride same padding 1  ##################################
# print('\n\n\nConv Test multiple filters multiple channels non unit stride same padding 1')
#  
# input = np.ones( (1, 3, 16, 16) )
#  
# model = Sequential()
# model.add( Conv2D( 2, (3,3), strides=(2,2), activation='square', padding='same', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
#  
# # filter shape (3, 3, 3, 1) and shape (1) bias
# weights = [ np.ones( (3, 3, 3, 2) ) , np.array([1, 1])]
# model.layers[0].set_weights(weights)
#  
# print(model.predict(input))
#  
#  
# ############### Conv Test batches multiple filters multiple channels non unit stride same padding 1  ##################################
# print('\n\n\nConv Test batches multiple filters multiple channels non unit stride same padding 1')
#  
# input = np.ones( (4, 3, 16, 16) )
#  
# model = Sequential()
# model.add( Conv2D( 2, (3,3), strides=(2,2), activation='square', padding='same', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
#  
# # filter shape (3, 3, 3, 1) and shape (1) bias
# weights = [ np.ones( (3, 3, 3, 2) ) , np.array([1, 1])]
# model.layers[0].set_weights(weights)
#  
# print(model.predict(input))

############### Conv Test multiple filters multiple channels same padding 1  ##################################
print('\n\n\nConv Test multiple filters multiple channels same padding 1')


from keras.utils.conv_utils import convert_kernel
 
K.set_image_data_format( 'channels_first' )
input = np.arange( 32 ).reshape( ( 1,2,4,4) )
 
model = Sequential()
model.add( Conv2D( 2, (3,3), strides=(1,1), activation='linear', padding='same', input_shape=input.shape[1:] ) )
model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
model.summary()
print( model.layers[0].get_config() )
 
# filter shape (3, 3, 2, 2) and shape (2) bias
weights =  np.arange( 36 ).reshape( ( 3,3,2,2 ) )
print( weights )

model.layers[0].set_weights( [weights, np.array( [0,1] ) ] )

t_input = np.arange(1, 37).reshape((1,1,6,6))

config = model.layers[0].get_config() 
#printing convolutional layers weights
if str(config.get("name")).find("conv2d") > -1:            
    weights = np.array(model.layers[0].get_weights())                        
    kernel_size = config.get("kernel_size")
    nomFilters = config.get("filters")
    nomFiltersPrevLayer = weights[0].shape[2]   
    for j in range(nomFilters):     
        for i in range(nomFiltersPrevLayer):
            sys.stdout.write( "-"+str(i)+"_"+str(j) +' \n' )           
            LayerWeights = weights[0][:,:,i,j]
            sys.stdout.write( str( LayerWeights.reshape(-1) ) )
            sys.stdout.write( '\n' )
        sys.stdout.write( '\n' )
  
