# 
# Used to generate baselines for other testcases
#

import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras import backend as K
import numpy as np
from prompt_toolkit import input


def square(x):
    return K.square(x)
a = keras.layers.Activation(square)
a.__name__ = 'square'
keras.utils.get_custom_objects().update( { 'square': keras.layers.Activation(square) } )


K.set_image_data_format( 'channels_first' )

# ############### Conv Test 1 ##################################
# print('\n\n\nConv Test 1 valid padding')
# 
# input = np.array( [ [ [ [ 1,1,1,1 ], [ 1,1,1,1 ],[ 1,1,1,1 ],[ 1,1,1,1 ] ] ] ] )
# 
# model = Sequential()
# model.add( Conv2D( 1, (3,3), strides=(1,1), activation='square', padding='valid', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
# 
# # print( model.layers[0].get_weights() )
# # print( model.layers[0].get_weights()[0].shape )
# 
# # filter shape (3, 3, 1, 1) and shape (1) bias
# weights = [np.array(  [[[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]]] ), np.array([1])]
# model.layers[0].set_weights(weights)
# 
# print(model.predict(input))
# 
# ############### Conv Test 2 ##################################
# print('\n\n\nConv Test 2 valid padding')
# 
# input = np.array( [ [ [ [ 1,2,2,1 ], [ 1,2,2,1],[ 1,2,2,1 ],[ 1,2,2,1 ] ] ] ] )
# 
# model = Sequential()
# model.add( Conv2D( 1, (3,3), strides=(1,1), activation='square', padding='valid', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
# 
# 
# # filter shape (3, 3, 1, 1) and shape (1) bias
# weights = [np.array(  [[[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]]] ), np.array([1])]
# model.layers[0].set_weights(weights)
# 
# print(model.predict(input))
# 
# ############### Conv Test 3 ##################################
# print('\n\n\nConv Test 3 valid padding')
# 
# input = np.array( [ [ [ [ 2,2,3,2 ], [ 2,1,2,3 ],[ 1,3,2,3 ],[ 2,2,1,2 ] ] ] ] )
# 
# model = Sequential()
# model.add( Conv2D( 1, (3,3), strides=(1,1), activation='square', padding='valid', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
# 
# 
# # filter shape (3, 3, 1, 1) and shape (1) bias
# weights = [np.array(  [[[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]]] ), np.array([1])]
# model.layers[0].set_weights(weights)
# 
# print(model.predict(input))
# 
# ############### Conv Test 4 ##################################
# print('\n\n\nConv Test 4 valid padding')
# 
# input = np.array( [ [ [ [ 1,3,3,2 ], [ 2,1,3,2 ],[ 1,3,2,1 ],[ 3,2,3,1 ] ] ] ] )
# 
# model = Sequential()
# model.add( Conv2D( 1, (3,3), strides=(1,1), activation='square', padding='valid', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
# 
# # filter shape (3, 3, 1, 1) and shape (1) bias
# weights = [np.array(  [[[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]]] ), np.array([1])]
# model.layers[0].set_weights(weights)
# 
# print(model.predict(input))
# 
# 
# 
# ############### Conv Test 5 ##################################
# print('\n\n\nConv Test 5 valid padding')
# 
# input = np.array( [ [ [ [ 3,2,3,2 ], [ 2,4,1,3 ],[ 2,2,1,3 ],[ 1,3,2,3 ] ] ] ] )
# 
# model = Sequential()
# model.add( Conv2D( 1, (3,3), strides=(1,1), activation='square', padding='valid', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
# 
# # filter shape (3, 3, 1, 1) and shape (1) bias
# weights = [np.array(  [[[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]]] ), np.array([1])] 
# model.layers[0].set_weights(weights)
# 
# print(model.predict(input))
# 
# ############### Conv Test strides 1 valid padding ##################################
# print('\n\n\nConv Test strides 1 valid padding')
# 
# input = np.ones((1, 1, 16, 16) )
# 
# model = Sequential()
# model.add( Conv2D( 1, (3,3), strides=(2,2), activation='square', padding='valid', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
# 
# # filter shape (3, 3, 1, 1) and shape (1) bias
# weights = [np.array(  [[[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]], [[[1]],[[1]],[[1]]]] ), np.array([1])] 
# model.layers[0].set_weights(weights)
# 
# print(model.predict(input))
# 
# ############### Conv Test multiple channels non unit stride valid padding 1  ##################################
# print('\n\n\nConv Test multiple channels non unit stride valid padding 1')
# 
# input = np.ones( (1, 3, 16, 16) )
# 
# model = Sequential()
# model.add( Conv2D( 1, (3,3), strides=(2,2), activation='square', padding='valid', input_shape=input.shape[1:] ) )
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
# ############### Conv Test multiple filters multiple channels non unit stride valid padding 1  ##################################
# print('\n\n\nConv Test multiple filters multiple channels non unit stride valid padding 1')
# 
# input = np.ones( (1, 3, 16, 16) )
# 
# model = Sequential()
# model.add( Conv2D( 2, (3,3), strides=(2,2), activation='square', padding='valid', input_shape=input.shape[1:] ) )
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
# ############### Conv Test batches multiple filters multiple channels non unit stride valid padding 1  ##################################
# print('\n\n\nConv Test batches multiple filters multiple channels non unit stride valid padding 1')
# 
# input = np.ones( (4, 3, 16, 16) )
# 
# model = Sequential()
# model.add( Conv2D( 2, (3,3), strides=(2,2), activation='square', padding='valid', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
# 
# # filter shape (3, 3, 3, 1) and shape (1) bias
# weights = [ np.ones( (3, 3, 3, 2) ) , np.array([1, 1])]
# model.layers[0].set_weights(weights)
# 
# print(model.predict(input))

# ############### Conv Test even filter  ##################################
# print('\n\n\nConv Test even filter')
#   
# input = np.arange( 16 ).reshape( ( 1,1,4,4 ) )
#   
# model = Sequential()
# model.add( Conv2D( 1, (2,2), strides=(1,1), activation='linear', padding='valid', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
#   
# # print( weights )
# print(model.layers[0].get_weights()[0])
# print(model.layers[0].get_weights()[0].shape)
# print(model.layers[0].get_weights()[1].shape)
# 
# weights = [ np.ones( ( 2, 2, 1, 1) ) , np.array([1])]
# model.layers[0].set_weights( weights )
# print(input)   
# print(model.predict(input))

############## Conv Test 1  CKKS##################################
# print('\n\n\nConv Test 1 valid padding CKKS')
#    
# input = np.array( [ [ [ [ 0.1,0.1,0.1,0.1 ], [ 0.1,0.1,0.1,0.1 ],[ 0.1,0.1,0.1,0.1 ],[ 0.1,0.1,0.1,0.1 ] ] ] ] )
# print(input.shape)
#    
# model = Sequential()
# model.add( Conv2D( 1, (3,3), strides=(1,1), activation='square', padding='valid', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
#    
# # filter shape (3, 3, 1, 1) and shape (1) bias
# weights = [np.zeros((3, 3, 1, 1))+0.1, np.array([0.1])]
# model.layers[0].set_weights(weights)
#    
# print(model.predict(input))
# ############### Conv Test 2  CKKS##################################
# print('\n\n\nConv Test 1 valid padding CKKS')
#    
# input = np.arange(32).reshape( 1,2,4,4 )
# input = 0.1*input 
# print(input.shape)
# print(input)
#    
# model = Sequential()
# model.add( Conv2D( 2, (3,3), strides=(1,1), activation='linear', padding='valid', input_shape=input.shape[1:] ) )
# model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
# model.summary()
#    
# # filter shape (3, 3, 1, 1) and shape (1) bias
# w = np.arange(36).reshape( 3,3,2,2 )
# w = w*0.1
# print(w)
# weights = [w, np.array([0.1,0.2])]
# model.layers[0].set_weights(weights)
#    
# print(model.predict(input))

############### Conv Test 2  CKKS##################################
print('\n\n\nConv Test 1 valid padding CKKS')
   
input = np.arange(32).reshape( 1,2,4,4 )
# input = np.ones( [1,2,3,3] )
input = 0.1*input
print(input.shape)
print(input)
   
model = Sequential()
model.add( Conv2D( 2, (3,3), strides=(1,1), activation='linear', padding='valid', input_shape=input.shape[1:] ) )
model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
model.summary()


     
  
# filter shape (3, 3, 1, 1) and shape (1) bias
w = np.arange(36,dtype=float).reshape( 3,3,2,2 )
# w = w*0.1
w[:,:,0,0] = np.arange( 0, 9,dtype=float ).reshape( 1,3,3 )*0.1
w[:,:,1,0] = np.arange( 9, 18,dtype=float ).reshape( 1,3,3 )*0.1
w[:,:,0,1] = np.arange( 18, 27,dtype=float ).reshape( 1,3,3 )*0.1
w[:,:,1,1] = np.arange( 27, 36,dtype=float ).reshape( 1,3,3 )*0.1
print(w)
for i in range(2):
    for j in range(2):
        print( w[:,:,i,j] )
        
        

weights = [w, np.array([0.1,0.2])]
model.layers[0].set_weights(weights)
   
print(model.predict(input))


