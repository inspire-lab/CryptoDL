from keras import backend as K

def layer_outputs( input_tensor, layers, x ):
    
    # check if it is a list or something else iterable
    try:
        _ = iter( layers )
        # dealing with a list or something
        outputs = [ l.output for l in layers  ]
    except Exception:
        outputs = [ layers.output ]         
    # evaluation function
    functor = K.function( [ input_tensor, K.learning_phase() ], outputs )   
     
    # run the function
    return functor( [ x, 0 ] )