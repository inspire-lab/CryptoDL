import keras
from keras import backend as K
import numpy as np
from keras.layers import SimpleRNN, Dense, Activation
from keras.models import Sequential
import sys
from subprocess import call


########## rnn1 #################
print( 'rnn1\n' )
input_dim = 3
time_steps = 2
units = 2
x = np.arange( time_steps * input_dim, dtype='float' )
x = x.reshape( (1,time_steps,input_dim) )
x = np.ones( (1,time_steps,input_dim) , 'float' )
print( x )

model = Sequential()
model.add( SimpleRNN( units, input_shape=x.shape[ 1: ], activation=None,return_sequences=True ) )
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

print( 'weigths')    
print( model.layers[ 0 ].get_weights()[ 0 ].shape )
print( model.layers[ 0 ].get_weights()[ 1 ].shape )
print( model.layers[ 0 ].get_weights()[ 2 ].shape )

weights = np.ones( input_dim * units, dtype='float' )
weights = weights.reshape( (input_dim, units)  ) 
recurrent_weights = np.ones( (units,units) )

bias = np.zeros( units, 'float' )
model.layers[ 0 ].set_weights( [ weights, recurrent_weights, bias ] )

print( model.predict( x ) )

    
########## rnn2 #################    
print( 'rnn2\n' )    
input_dim = 3
time_steps = 3
units = 2
x = np.arange( time_steps * input_dim, dtype='float' )
x = x.reshape( (1,time_steps,input_dim) )
x = np.ones( (1,time_steps,input_dim) , 'float' )

x = np.array( [ [ [1,1,1], [0,0,0], [0,0,0] ] ] )
print( x )

model = Sequential()
model.add( SimpleRNN( units, input_shape=x.shape[ 1: ], activation=None,return_sequences=True ) )
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

weights = np.ones( input_dim * units, dtype='float' )
weights = weights.reshape( (input_dim, units)  ) 
# recurrent_weights = np.arange( units * units ).reshape( (units,units) )
recurrent_weights = np.array( [[ 0,2 ],[ 1 ,3 ]] )

print( weights )
print( recurrent_weights )

bias = np.zeros( units, 'float' )
model.layers[ 0 ].set_weights( [ weights, recurrent_weights, bias ] )

print( model.predict( x ) )    

########## rnn3 #################    
print( 'rnn3\n' )    
input_dim = 3
time_steps = 3
units = 2
x = np.arange( time_steps * input_dim, dtype='float' )
x = x.reshape( (1,time_steps,input_dim) )
# x = np.ones( (1,time_steps,input_dim) , 'float' )

print( x )

model = Sequential()
model.add( SimpleRNN( units, input_shape=x.shape[ 1: ], activation=None,return_sequences=True ) )
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

weights = np.array( [ [ 0, 3 ],[ 1 , 4 ], [ 2 ,5 ] ] )
recurrent_weights = np.zeros( (units,units) )

print( weights )
print( recurrent_weights )

bias = np.zeros( units, 'float' )
model.layers[ 0 ].set_weights( [ weights, recurrent_weights, bias ] )

print( model.predict( x ) )    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
