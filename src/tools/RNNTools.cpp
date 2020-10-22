#include <cstdlib>
#include <iostream>
#include <string>
#include "FileSystemTools.h"
#include "DataReaders.h"
#include "SystemTools.h"


std::pair<std::vector<float>,std::vector<int8_t>> getEmbeddings( const std::string& modelFile, int maxlen, int maxwords, std::string dataset, std::string outDir ){

	/* Setup the python code to call the following function:
	 *
	 * def dump_embeddings( model_file, output_dir, maxlen = 200, maxwords=20000, dataset='imdb', exit_on_completion=True, batch_size=128):
	 *    '''
	 *   Dumps the output of the embedding layer of the `model_file` to the `output_dir`. Embeddings are created
	 *    for the test_data. Writes binary data. Embeddings are written as `float32`, labels as `int8`.
	 *
	 *    Args:
	 *        model_file: string path to the moodel file
	 *       output_dir: string path to the output directory
	 *        maxlen: maximum length of the sequence to be embedded
	 *        maxwords: number of top words to use
	 *        dataset: one of the following: 'imdb'
	 *        exit_on_completion: call `exit()` on completion or failure
	 *       batch_size: batch size for embedding computation
	 *
	 *   Exit codes:
	 *        0 : all good
	 *        1 : unsupported dataset
	 *        2 : error loading model
	 *        3 : model does not have an embedding layer
	 *    '''
	 */
	static const std::string pythonScript =
		"\"import sys\n"
		"from kalypso.rnn_tools import dump_embeddings\n"
		"dump_embeddings( sys.argv[ 1 ], sys.argv[ 2 ], maxlen=int(sys.argv[ 3 ]), maxwords=int(sys.argv[ 4 ]), dataset=sys.argv[ 5 ] ) \"";

	// set default outdir
	if ( outDir == "" )
		outDir = getDirectory( modelFile );

	std::string command = pythonScript + " " +  modelFile + " " + outDir + " " + std::to_string( maxlen ) + " " + std::to_string( maxwords ) + " " + dataset;

	static const int PY_EXIT_SUCCESS = 0;
	static const int PY_EXIT_UNSUPPORTED = 1;
	static const int PY_EXIT_LOADING_ERORR = 2;
	static const int PY_EXIT_NO_EMBEDDING = 3;


	// run python code
	int exitCode = executePython( command );
	if ( exitCode  != PY_EXIT_SUCCESS  ){
		std::cerr <<  "embeddign extraction failed" <<   std::endl;
		switch ( exitCode ) {
			case PY_EXIT_UNSUPPORTED:
				std::cerr <<  "unsupported dataset " << dataset <<  std::endl;
				break;
			case PY_EXIT_LOADING_ERORR:
				std::cerr <<  "error loading model " << modelFile <<  std::endl;
				break;
			case PY_EXIT_NO_EMBEDDING:
				std::cerr <<  "model has no embedding layer" << std::endl;
				break;
			default:
				break;
		}
		exit( exitCode );
	}

	// build filenames that were used by the python code
	std::string embeddingFile = outDir + "/x_" + getFileName( modelFile );
	std::string labelFile = outDir + "/y_" + getFileName( modelFile );
	// read the binary data
	std::vector<float> embeddings = readFloat32FromBinary( embeddingFile );
	std::vector<int8_t> labels = readInt8FromBinary( labelFile );
	// wrap in a pair and return
	std::pair<std::vector<float>,std::vector<int8_t>> ret( embeddings, labels  );
	return ret;
}
