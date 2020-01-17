import kalypso
import keras

# load the pretrained model
m = keras.models.load_model( 'mnistCNN.h5' )
# extract the model to the weights folder
kalypso.export_weights( m, 'weights' )


