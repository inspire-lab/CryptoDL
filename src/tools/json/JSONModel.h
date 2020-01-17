/*
 * JSONModel.h
 *
 *  Created on: Apr 22, 2019
 *      Author: agustin
 */

#ifndef JSON_JSONMODEL_H_
#define JSON_JSONMODEL_H_


#include "../../architecture/Model.h"
#include "../../architecture/Layer.h"
#include "json.hpp"
#include <unordered_map>
#include <exception>
#include <stdexcept>
#include <memory>
#include "../FileSystemTools.h"
#include "../SystemTools.h"
using json = nlohmann::json;

//custom exception for invalid layer types
class InvalidLayer: public std::exception {
public:
	//creates exception message
	InvalidLayer( const std::string& layerType ) :
			m_msg( layerType + " is not supported." ) {
	}
	const char * what() const throw () {
		return m_msg.c_str();
	}
private:
	std::string m_msg;
};

//checks whether a layer is in current implementation of architecture
//TODO: make a global container and define in main to avoid reinitializing
//essentially memoize the valid classes - useful if we ever want continually running program i.e. service
bool isValidLayerClass( std::string layer );

//checks string from json object to return valid enum type
inline PADDING_MODE grabPadding( std::string pad ) {
	return ( pad == "same" ) ? PADDING_MODE::SAME : PADDING_MODE::VALID;
}

//checks input from json object to return correct activation pointer
template<class ValueType>
ActivationP<ValueType> grabActivation( std::string act, std::map<std::string, ActivationP<ValueType>>& map=std::map<std::string, ActivationP<ValueType>>() ) {

	// check if we have replacement
	try {
		return map.at( act );
	}
	catch (const std::exception& e)	{
		std::cerr << e.what() << std::endl;
	}
	if ( act == "square" )
		return SquareActivation<ValueType>::getSharedPointer();
	if ( act == "linear" )
		return LinearActivation<ValueType>::getSharedPointer();
	if ( act == "relu" )
//		return ReluActivation<ValueType>::getSharedPointer();
		throw std::runtime_error( "relu currently disabled" ); // FIXME enable relu for none HE types
	std::cerr << "Error: " << act << " is not currently implemented." << std::endl;
	exit( -1 );
}






// TODO: i am not happy with the design of this. Ideally the layers themselfs would hold the code on how to load them
// and register the with a loader class. but this'll do for now
template<class ValueType, class WeightType, class DataTensorType, class WeightTensorType>
class ModelLoader {
public:

	ModelLoader( TensorFactoryP<ValueType> dtf, TensorFactoryP<WeightType> wtf,	std::map<std::string, ActivationP<ValueType>> activationMap = std::map<std::string, ActivationP<ValueType>>(), bool skipUnknown=true )
		: mActivationMap( activationMap ), mDtf( dtf ), mWtf( wtf ), mSkipUnknown( skipUnknown ) {
	}

	/**
	 * @brief Takes a filename. Extracts the weights and the architecture. If the output files exist already
	 * nothing is done.
	 *
	 * This function runs some python code to do the actual extraction.
	 */
	Model<ValueType, WeightType, DataTensorType, WeightTensorType> load( std::string filePath, std::string outdir, int batchSize=1 ){
		std::string fname = getFileName( filePath );
		std::string weightDir = outdir + "/weights/";
		std::string jsonFile = outdir + "/" + fname + ".json" ;
		std::cout << jsonFile << std::endl;

		// check if we need to do pyhton parsing
		// TODO make it callable on its own
		if( ! fileExists( weightDir ) || ! fileExists( jsonFile )  ){
			std::string cmd =  python_code + " " + filePath + " " + outdir + " " + jsonFile;
			std::cout << "Running python exporting script....";
			if ( executePython( cmd ) != EXIT_SUCCESS )
				exit( 1 );
			std::cout << "Done" << std::endl;
		}

		// load the model
		auto model = fromFile( jsonFile, batchSize );
		// load weights
		model.loadWeights( weightDir );

		return model;

	}

	Model<ValueType, WeightType, DataTensorType, WeightTensorType> fromFile( std::string filename, int batchSize=1, MemoryUsage usage = MemoryUsage::greedy ) {
		//read json file into json object
		json modelConfig;
		try {
			std::ifstream inputFile( filename );
			inputFile >> modelConfig;
			inputFile.close();
		} catch ( const std::ifstream::failure& e ) {
			std::cerr << e.what() << std::endl;
		}

		//ensure correct info
		assert( modelConfig [ "backend" ] == "tensorflow" );

		//grab layers
		json layers = modelConfig [ "config" ] [ "layers" ];
		Model<ValueType, WeightType, DataTensorType, WeightTensorType> model( usage, mDtf.get(), mWtf.get() );

		bool isFirst = true;
		bool isFirstKnown = true;
		//for each layer, find attributes
		TensorP<ValueType> input = nullptr;
		for ( auto& layerJson : layers ) {
			std::string layerType = layerJson [ "class_name" ];
				LayerP<ValueType, WeightType, DataTensorType, WeightTensorType> layer;

				std::cout << layerType << std::endl;
				if( isFirst ){
					input = grabInputTensor( layerJson [ "config" ],  layerType, batchSize );
					isFirst = false;
				}

				//find correct layer usage
				if ( layerType == "Conv2D" )
					layer = grabConv2D( layerJson [ "config" ] );
				else if ( layerType == "Flatten" )
					layer = grabFlatten( layerJson [ "config"]  );
				else if ( layerType == "Dense" )
					layer = grabDense( layerJson [ "config" ] );
				else if ( layerType == "SimpleRNN" )
					layer = grabRNN( layerJson [ "config" ] );
				else if ( layerType == "AveragePooling2D" )
					layer = grabAveragePooling( layerJson [ "config" ] );
				else if ( layerType == "ZeroPadding2D" )
					layer = grabZeroPadding2D( layerJson [ "config" ] );
				else if( mSkipUnknown )
					continue;
				else
					throw InvalidLayer( layerType );

				if( isFirstKnown ){
					layer->input( input );
					isFirstKnown = false;
					layer->output( mDtf->create( layer->outputShape() ) );
				}

				//add layer
				model.addLayer( layer );

		}

		return model;
	}


	LayerP<ValueType, WeightType, DataTensorType, WeightTensorType> grabConv2D(	json conv2d) {
		//grab layer attributes
		std::string name = conv2d [ "name" ];
		ActivationP<ValueType> act = grabActivation<ValueType>(	conv2d [ "activation" ], mActivationMap );
		int noFilters = conv2d [ "filters" ];
		int filterSize = conv2d [ "kernel_size" ] [ 0 ];
		int stride = conv2d [ "strides" ] [ 0 ];
		PADDING_MODE pad = grabPadding( conv2d [ "padding" ] );
		//is not the first layer in the model
		return std::make_shared<
				Convolution2D<ValueType, WeightType, DataTensorType,
						WeightTensorType>>( name, act, noFilters, filterSize,stride, pad );
	}

	LayerP<ValueType, WeightType, DataTensorType, WeightTensorType> grabDense(	json dense) {
		//grab layer attributes
		std::string name = dense [ "name" ];
		ActivationP<ValueType> act = grabActivation<ValueType>(	dense [ "activation" ], mActivationMap );
		int noNeurons = dense [ "units" ];
		//layer is not first in model, no need to grab input dimensions
		return std::make_shared<
				Dense<ValueType, WeightType, DataTensorType, WeightTensorType>>(
				name, act, noNeurons );
	}

	LayerP<ValueType, WeightType, DataTensorType, WeightTensorType> grabFlatten( json flatten) {
		//grab layer attribute
		std::string name = flatten [ "name" ];
		//layer is not first layer in model
		return std::make_shared<Flatten<ValueType, WeightType, DataTensorType, WeightTensorType>>(	name );
	}

	LayerP<ValueType, WeightType, DataTensorType, WeightTensorType> grabZeroPadding2D( json config) {
		//grab layer attribute
		std::string name = config [ "name" ];
		int padding = config[ "padding" ][ 0 ][ 0 ];
		//layer is not first layer in model
		return std::make_shared<ZeroPadding2D<ValueType, WeightType, DataTensorType, WeightTensorType>>(name, padding );
	}

	LayerP<ValueType, WeightType, DataTensorType, WeightTensorType> grabAveragePooling(	json config) {
		//grab layer attributes
		std::string name = config[ "name" ];
		int filterSize = config[ "kernel_size" ] [ 0 ];
		int stride = config[ "strides" ] [ 0 ];
		PADDING_MODE pad = grabPadding( config[ "padding" ] );
		//is not the first layer in the model
		return std::make_shared<AveragePooling<ValueType, WeightType, DataTensorType,
						WeightTensorType>>( name, filterSize,stride, pad );
	}

	LayerP<ValueType, WeightType, DataTensorType, WeightTensorType> grabRNN( json config) {
		//grab layer attributes
		std::string name = config[ "name" ];
		ActivationP<ValueType> act = grabActivation<ValueType>(	config[ "activation" ], mActivationMap );
		int noNeurons = config[ "units" ];
		bool return_sequences = config[ "return_sequences" ];

		//layer is not first in model, no need to grab input dimensions
		return std::make_shared<
				RNN<ValueType, WeightType, DataTensorType, WeightTensorType>>(
				name, act, noNeurons, return_sequences );
	}


private:
	std::map<std::string, ActivationP<ValueType>> mActivationMap;
	TensorFactoryP<ValueType> mDtf;
	TensorFactoryP<WeightType> mWtf;
	bool mSkipUnknown;

	/**
	 * python script that exports the model architecture and weight.
	 * the kalypso module needs to be installed
	 */
	const std::string python_code =
	"\"from kalypso import model_exporter \n"
	"import sys \n"
	"model_exporter( sys.argv[ 1 ], sys.argv[ 2 ], sys.argv[ 3 ] ) \"";



	TensorP<ValueType> grabInputTensor( json config, std::string layerType = "", int batchSize=1 ) {
		//grab input shape and 'clean' since shapes in keras can be null, we convert them to 1 to avoid issues
		std::vector<size_t> shapeVector;

		// embedding layers need some special handling becasue we don't do the embedding. We want to get the outputshape
		// of the embedding layer.
		if( layerType == "Embedding" ){
			// (batch_size, sequence_length, output_dim)
			shapeVector.push_back( batchSize ); 
			shapeVector.push_back( config[ "input_length" ] );
			shapeVector.push_back( config[ "output_dim" ] );

		}else{

			auto shape = config [ "batch_input_shape" ];
			for ( auto& el : shape ){
				if ( el.is_null() )
					el = 1;
				shapeVector.push_back( el );
			}
		}
		Shape inputShape( shapeVector );
		TensorP<ValueType> input = mDtf->create( inputShape );
		return input;
	}




};


#endif /* JSON_JSONMODEL_H_ */
