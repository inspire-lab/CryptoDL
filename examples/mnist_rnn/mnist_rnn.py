import kalypso
import keras


# add the activaition to the custom object dict by hand
def polyTanhX(x):
    return -0.00163574303018748*x**3 + 0.249476365628036*x
b = keras.layers.Activation(polyTanhX)
b.__name__ = 'tanh_aprox'
keras.utils.get_custom_objects().update( { 'tanh_aprox': b } )


# load the pretrained model
m = keras.models.load_model( 'mnistRNN.h5' )
# extract the model to the weights folder
kalypso.export_weights( m, 'weights' )


