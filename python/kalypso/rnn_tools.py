from tensorflow import keras
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import imdb
from kalypso import actFunctions
from tensorflow.keras.preprocessing import sequence
import sys
import os

supported_datasets = [ 'imdb', 'imdb_glove' , 'amazon_movies', 'amazon_books' ]


def dump_embeddings( model_file, output_dir, maxlen=200, maxwords=20000, dataset='imdb', exit_on_completion=True, batch_size=128 ):
    '''
    Dumps the output of the embedding layer of the `model_file` to the `output_dir`. Embeddings are created
    for the test_data. Writes binary data. Embeddings are written as `float32`, labels as `int8`.
    
    Args:
        model_file: string path to the moodel file
        output_dir: string path to the output directory
        maxlen: maximum length of the sequence to be embedded
        maxwords: number of top words to use
        dataset: one of the following: 'imdb'
        exit_on_completion: call `exit()` on completion or failure
        batch_size: batch size for embedding computation
        
    Exit codes:
        0 : all good
        1 : unsupported dataset
        2 : error loading model 
        3 : model does not have an embedding layer
    '''

    # is the dataset supported
    if not dataset in supported_datasets and exit_on_completion:
        exit( 1 )

    # build output filename
    out_file = model_file.split( '/' )[ -1 ]

    # return fast if the files exist
    if os.path.exists( os.path.join( output_dir, 'x_' + out_file ) ) and  os.path.exists( os.path.join( output_dir, 'x_' + out_file ) ):
        print( 'files already exist' )
        return

    # load the model
    try:
        model = keras.models.load_model( model_file )
    except Exception as e:
        sys.stderr.write( e )
        if exit_on_completion:
            exit( 2 )
        return

    # make sure the first layer is an embedding layer
    if 'embedding' not in model.layers[ 0 ].get_config()[ 'name' ]:
        sys.stderr.write( 'model does not have an embedding layer' )
        if exit_on_completion:
            exit( 3 )
        return

    # load data
    if dataset == 'imdb':
        ( x_train, y_train ), ( x_test, y_test ) = imdb.load_data( num_words=maxwords )

    print( 'Loaded dataset with {} training samples, {} test samples'.format( len( x_train ), len( x_test ) ) )

    # pad data
    x_test = sequence.pad_sequences( x_test, maxlen=maxlen )
    print( 'x_test', x_test.shape )

    # input placeholder
    inp = model.input
    # output of the embedding layer
    outp = model.layers[ 0 ].output
    # evaluation function
    functor = K.function( [inp, K.learning_phase()], [outp] )

    # build batches and run through the embedding layer
    with open( os.path.join( output_dir, 'x_' + out_file ), 'wb' ) as fx :
        # iterate over all batches
        i = 0
        while( i * batch_size <= x_test.shape[ 0 ] ):
            # build batch
            batch = x_test[ batch_size * i : min( batch_size * ( i + 1 ), x_test.shape[ 0 ] ) ]
            i += 1

#             print( batch.shape )
            # execute function
            embeddings = functor( [batch, 0] )[0]
#             print( len(embeddings) )
            print( embeddings.shape )

            # save to file
            fx.write( embeddings.astype( np.float32 ).tobytes() )

    # write labels
    with open( os.path.join( output_dir, 'y_' + out_file ), 'wb' ) as fy:
        fy.write( y_test.astype( np.int8 ).tobytes() )

