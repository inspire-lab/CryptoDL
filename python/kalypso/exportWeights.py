import os
import tempfile
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import datasets
import numpy as np
import sys

to_sysout = False


def SavingOriginalWeights( model, path ):
    if not to_sysout:
        print( 'saving to ', path )

    # create the directory
    try:
        os.mkdir( path )
    except FileExistsError:
        # dir exists. can be ignored
        pass
    layers = model.layers
    for i in range( len( layers ) ):
        config = layers[i].get_config()
        print( config.get( "name" ) )
        # printing convolutional layers weights
        if str( config.get( "name" ) ).find( "conv2d" ) > -1:
            weights = np.array( layers[i].get_weights() )
            kernel_size = config.get( "kernel_size" )
            nomFilters = config.get( "filters" )
            nomFiltersPrevLayer = weights[0].shape[2]
            for i in range( nomFiltersPrevLayer ):
                for j in range( nomFilters ):
                    LayerWeights = weights[0][:, :, i, j]
                    LayerWeights = np.append( LayerWeights, weights[1][j] )
                    strName = os.path.join( path, config.get( "name" ) + "-" + str( i ) + "_" + str( j ) + ".txt" )
                    if not to_sysout:
                        np.savetxt( strName, LayerWeights, newline=' ', fmt='%f' )
                    else:
                        print( config.get( "name" ) + "-" + str( i ) + "_" + str( j ), LayerWeights )

        # printing fully connected layers weights
        elif str( config.get( "name" ) ).find( "dense" ) > -1:
            weights = layers[i].get_weights()
            strName = config.get( "name" )
            nomNeurons = config.get( "units" )
            nomNeuronsPrevLayer = weights[0].shape[0]
            for j in range( nomNeurons ):
                LayerWeights = weights[0][:, j]
                strName = os.path.join( path, config.get( "name" ) + "_" + str( j ) + ".txt" )
                if not to_sysout:
                    np.savetxt( strName, LayerWeights, newline=' ', fmt='%f' )

            strName = os.path.join( path, config.get( "name" ) + "_bias" + ".txt" )
            if not to_sysout:
                np.savetxt( strName, weights[1], newline=' ', fmt='%f' )

        elif str( config.get( "name" ) ).find( "rnn" ) > -1:
            weights = layers[i].get_weights()
            strName = config.get( "name" )
            nomNeurons = config.get( "units" )
            nomNeuronsPrevLayer = weights[0].shape[0]
            for j in range( nomNeurons ):
                LayerWeights = weights[0][:, j]
                strName = os.path.join( path, config.get( "name" ) + "_" + str( j ) + ".txt" )
                if not to_sysout:
                    np.savetxt( strName, LayerWeights, newline=' ', fmt='%f' )

                # save reccurrent kernel
                recurrent_weights = weights[1][:, j]
                strName = os.path.join( path, config.get( "name" ) + "_recurrent_" + str( j ) + ".txt" )
                if not to_sysout:
                    np.savetxt( strName, recurrent_weights, newline=' ', fmt='%f' )

            strName = os.path.join( path, config.get( "name" ) + "_bias" + ".txt" )
            if not to_sysout:
                np.savetxt( strName, weights[2], newline=' ', fmt='%f' )

        # skipping activation and pooling layers
        elif str( config.get( "name" ) ).find( "pooling" ) > -1 or \
        str( config.get( "name" ) ).find( "activation" ) > -1:
            continue


def write_header( weights, f ):
    """
    Writes header to the file. File must be open
    """

    # write version header
    f.write( np.ubyte( 0 ).tobytes() )
    # write reserved field
    f.write( np.ubyte( 0 ).tobytes() )
    # write the number of dimensions
    f.write( np.uint32( len( weights.shape ) ).tobytes() )
    # write the dimension info
    for d in weights.shape:
        f.write( np.uint32( d ).tobytes() )
    # write datatype info (currently it is only float32
    f.write( np.ubyte( 1 ).tobytes() )


def save_weights_v2( model, path ):
    """
    
    WIP
    
    
    binary file format:
    
    version 0:
        [ 1byte: version ][ 1byte: reserved ][ 4byte: no dimension n ][ n*4 byte: dimensions ][ 1byte: datatype ][ ...data... ]
    datatypes:
        version: unsigned char
        no dimensions n: unsigned char
        dimensions: 32bit unsignet integer 
        datatype: unsigned char
    meaning of the fields:
        version:
            what version of the file is being read headerfields might change depending on the version. 
            current version: 0 
        reserved:
            currently unused
        no dimensions:
            unsigned integer that indicates the number of dimesions of the stored tensor
        dimensions:
            a number of unsigned ints that indicate the number of elements for each tensor dimension. the number of unsigned ints
            is given by the previous field
        datatype:
            the datatype of the tensor. supported types:
            0: int32
            1: float32
        data:
            the actual payload. the size should be the product of all dimensions * size_of( dataype ) in bytes
    
    
    
    """
    print( 'saving (v2) to ', path )
    for layer in model.layers:
        config = layer.get_config()
        # save convolutional layers weights
        class_name = config[ 'class_name' ]
        if class_name == 'Conv2D' :
            weights = layer.get_weights()
            kernel_size = config.get( 'kernel_size' )
            nomFilters = config.get( 'filters' )
            nomFiltersPrevLayer = weights[0].shape[2]
            with open( os.path.join( path , config[ 'name' ] + '.bin' ), 'wb' ) as f:
                write_header( weights[0], f )
                for i in range( nomFiltersPrevLayer ):
                    for j in range( nomFilters ):
                        f.write( weights[0][:, :, i, j].reshape( -1 ).astype( np.float32 ).tobytes() )
            with open( os.path.join( path , config[ 'name' ] + '_bias.bin' ), 'wb' ) as f:
                write_header( weights[1], f )
                f.write( weights[1].astype( np.float32 ).tobytes() )

        # saving fully connected layers weights
        elif "dense" in config.get( "name" ):
            weights = layer.get_weights()
            strName = config.get( "name" )
            nomNeurons = config.get( "units" )
            nomNeuronsPrevLayer = weights[0].shape[0]
            with open( os.path.join( path , config[ 'name' ] + '.bin' ), 'wb' ) as f:
                write_header( weights[0], f )
                for j in range( nomNeurons ):
                    f.write( weights[0][:, j].reshape( -1 ).astype( np.float32 ).tobytes() )

            with open( os.path.join( path , config[ 'name' ] + '_bias.bin' ), 'wb' ) as f:
                write_header( weights[1], f )
                f.write( weights[1].astype( np.float32 ).tobytes() )

        elif str( config.get( "name" ) ).find( "rnn" ) > -1:
            weights = layers[i].get_weights()
            strName = config.get( "name" )
            nomNeurons = config.get( "units" )
            nomNeuronsPrevLayer = weights[0].shape[0]
            with open( os.path.join( path , config[ 'name' ] + '.bin' ), 'wb' ) as f:
                write_header( weights[ 0 ], f )
                for j in range( nomNeurons ):
                    LayerWeights = weights[0][:, j]
                    f.write( weights[0][:, j].reshape( -1 ).astype( np.float32 ).tobytes() )

            # save reccurrent kernel
            with open( os.path.join( path , config[ 'name' ] + '_recurrent.bin' ), 'wb' ) as f:
                write_header( weights[ 1 ], f )
                f.write( weights[1][:, j].reshape( -1 ).astype( np.float32 ).tobytes() )

            with open( os.path.join( path , config[ 'name' ] + '_bias.bin' ), 'wb' ) as f:
                write_header( weights[2], f )
                f.write( weights[2].astype( np.float32 ).tobytes() )


def export_weights( model, path, format=1 ):
    if format == 1:
        SavingOriginalWeights( model, path )
    elif format == 2:
        save_weights_v2( model, path )
    else:
        raise RuntimeError( 'unsoprted format' )


if __name__ == '__main__':
    from . import actFunctions
    from .keras.Layers import custom_objects
    # load model
    model = keras.models.load_model( sys.argv[ 1 ], custom_objects=custom_objects )
    SavingOriginalWeights( model, sys.argv[ 2 ] )

