#include <cstdlib>
#include <iostream>
#include <string>
#include "FileSystemTools.h"
#include "DataReaders.h"
#include "SystemTools.h"
#include <algorithm>
#include "RNNTools.h"
#include "IOStream.h"


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


Embedding::Embedding(uint embeddingDim, uint inputDim, const std::string& file ) : embeddingDim( embeddingDim ), inputDim( inputDim ) {
        std::vector<float> flatMatrix = readFloat32FromBinary( file );
	std::cout << "read " << flatMatrix.size() << " values from file" << std::endl;
        // sanity check
		if ( flatMatrix.size() % embeddingDim != 0 )
			throw std::logic_error( "invalid embedding matrix shape" );

        for ( size_t i = 0; i <= flatMatrix.size() / embeddingDim; ++i ){
            embeddingMatrix.push_back( std::vector<float>() );
            for ( size_t j = 0; j < embeddingDim; ++j ){
                embeddingMatrix[ i ].push_back( flatMatrix[ ( i * embeddingDim ) + j ] );
            }
//	std::cout << embeddingMatrix[i] <<std::endl;
        }
	std::cout << "created " << embeddingMatrix.size() <<  " embeddings" << std::endl;

}


 std::vector<float> Embedding::embed( const std::vector<std::vector<int>>& idx, int batchSize ){
	std::vector<float> embeddings( idx.size() * inputDim * embeddingDim );

	// TODO add batchsize and use copy
	size_t c = 0; // counter
	for ( size_t i = 0; i < idx.size(); ++i )
		for ( size_t j = 0; i < inputDim; ++i )
			for( float e: embeddingMatrix[ idx[ i ][ j ] ] )
				embeddings[ c++ ] = e;

	return embeddings;
 }

  std::vector<float> Embedding::embed( const std::vector<int>& idx, int batchSize ){
	// a single instance has inputDim elements
	batchSize = batchSize == -1 ? idx.size() : batchSize;
	if ( batchSize > idx.size() ){
		std::cerr << "batch size of " << batchSize << " too large for embedding with imput dim: " << inputDim << " and " << idx.size() << " inputs " << std::endl;
		exit( 1 );
	}
	std::cout << "creating embedding vector, size: " << batchSize << std::endl;
	std::vector<float> embeddings( batchSize * inputDim * embeddingDim, 0.0 );
	std::cout << embeddings.size() << " done" << std::endl;

	auto target = embeddings.begin();
	for ( size_t i = 0; i < inputDim * batchSize; ++i ){
		//std::cout << "copying: " << i << "/" << batchSize << " " << std::distance( embeddings.begin(), target ) << "/" << embeddings.size() << std::endl;
		//std::cout << idx[i] << "/" << embeddingMatrix.size() << std::endl;
		auto& temp = embeddingMatrix[ idx[ i ] ];
		//std::cout << temp[0] << std::endl;
		//std::cout << "word index " << idx[ i ] << "," << temp.size() << std::endl;
		auto start = temp.begin();
		//std::cout << "one last thing" << std::endl;
		target = std::copy( start, start + embeddingDim, target );
	}
	return embeddings;
 }


std::vector<float> Embedding::embed_truncated( const std::vector<int>& idx, int trunc, int batchSize ) {
	// a single instance has inputDim elements
	batchSize = batchSize == -1 ? idx.size() : batchSize;
	if ( batchSize > idx.size() ){
		std::cerr << "batch size of " << batchSize << " too large for embedding with imput dim: " << inputDim << " and " << idx.size() << " inputs " << std::endl;
		exit( 1 );
	}
	std::cout << "creating embedding vector, size: " << batchSize << std::endl;
	std::vector<float> embeddings( batchSize * ( inputDim - trunc ) * embeddingDim, 0.0 );
	std::cout << embeddings.size() << " done" << std::endl;

	auto target = embeddings.begin();
	for ( size_t batch = 0; batch < batchSize; ++batch )
		for ( size_t i = trunc; i < inputDim; ++i ){
			auto& temp = embeddingMatrix[ idx[ ( batch * inputDim ) + i ] ];
			auto start = temp.begin();
			target = std::copy( start, start + embeddingDim, target );
		}
	return embeddings;
 }