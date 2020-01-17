import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, Flatten, Dense
from keras import backend as K
import numpy as np
from keras.models import model_from_json
from prompt_toolkit import input
import sys

# to avoid attribute error, extend activation class
class Square(Activation):
    def __init__(self, activation, **kwargs):
        super(Square, self).__init__(activation, **kwargs)
        self.__name__ = 'square'

def square(x):
    return K.square(x)

#keras.utils.get_custom_objects().update( { 'square': keras.layers.Activation(square) } )
keras.utils.get_custom_objects().update( { 'square': Square(square) } )

K.set_image_data_format( 'channels_first' )

input = np.array( [ [ [ [ 1,1,1,1 ], [ 1,1,1,1 ],[ 1,1,1,1 ],[ 1,1,1,1 ] ] ] ] )

# define model
model = Sequential()
model.add( Conv2D( 1, (3,3), strides=(1,1), activation='square', padding='same', input_shape=input.shape[1:] ) )
model.add(Flatten())
model.add(Dense(32))
model.compile( 'adam', 'mean_squared_error', ['accuracy'] )
model.summary()

# save model as json
model_json = model.to_json()
with open("../json/model.json", "w") as json_file:
    json_file.write(model_json)
