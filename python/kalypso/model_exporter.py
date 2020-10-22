import os
from tensorflow import keras
from .exportWeights import export_weights
from .keras.Layers import custom_objects


def model_exporter( model_file, output_dir, json_file ):
    """Exporter for keras models
    
    # Arguments
        model_file: Path to the saved model file
        output_dir: Path to the directory where all the ouput put files will 
            be saved. The folders will be created if they don't exist
        json_file: Filename of the file the json string will be written to.
            It will be placed in `output_dir`
    
    # Notes:
        If you are using custom activation functions they need to be loaded
        into the custom objects dict provided by keras.
    
    """

    # create output folder if it does not exist
    try:
        import pathlib
        pathlib.Path( output_dir + '/weights/' ).mkdir( parents=True, exist_ok=True )
    except:
	    # python 2 backup
	    if not os.path.exists( directory ):
             os.makedirs( directory )

    # load the model
    model = keras.models.load_model( model_file, custom_objects=custom_objects )
    # export the weights
    export_weights( model, output_dir + '/weights/' )
    # get json config
    json_string = model.to_json()
    # save json to file
    with open( json_file, 'w' ) as f:
        f.write( json_string )
