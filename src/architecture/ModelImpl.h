/*
 * ModelImpl.h
 *
 *  Created on: Feb 20, 2019
 *      Author: robert
 */

#ifndef ARCHITECTURE_MODELIMPL_H_
#define ARCHITECTURE_MODELIMPL_H_


#include <vector>
#include <chrono>
#include <ctime>
#include <iostream>
#include "Layer.h"
#include "Tensor.h"
#include "PlainTensor.h"
#include "HEBackend/helib/HELIbCipherText.h"

//TODO: have it do something
enum MemoryUsage {
	lazy = 1,
	greedy = 2,
	free_after_use = 4,
	pre_convert_weights = 8,

};


inline MemoryUsage operator|( MemoryUsage x, MemoryUsage y ){
	return static_cast<MemoryUsage>( static_cast<int>( x ) | static_cast<int>( y ) );
}

inline MemoryUsage operator&( MemoryUsage x, MemoryUsage y ){
	return static_cast<MemoryUsage>( static_cast<int>( x ) & static_cast<int>( y ) );
}


template<class ValueType, class WeightType, class DataTensorType=Tensor<ValueType>, class WeightTensorType=Tensor<WeightType>, class ConvertType=float>
class Model {
public:

	Model( MemoryUsage usage, TensorFactory<ValueType>* dtf, TensorFactory<WeightType>* wtf, WeightConverter<WeightType,ConvertType>* weightConverter )
		: mLayers( { } ), mUsage( usage ), mDataFactory( dtf ), mWeightFactory( wtf ), mWeightConverter( weightConverter ) {
		if( (usage & MemoryUsage::pre_convert_weights) && weightConverter == nullptr ){
			std::cerr << "Can not create with with pre weight conversion and no converter" << std::endl;
			std::exit( 1 );
		}
	}

	Model( MemoryUsage usage, TensorFactory<ValueType>* dtf, TensorFactory<WeightType>* wtf )
		: Model<ValueType, WeightType, DataTensorType, WeightTensorType, ConvertType>( usage, dtf, wtf, nullptr ) {
	}

	//sequentially adds passed-in layer to container
	void addLayer( LayerP<ValueType, WeightType, DataTensorType, WeightTensorType> layer ) {
		if ( mLayers.empty() && layer->input() == NULL ) {
			std::cerr << "first layers need an input shape" << std::endl;
			exit( 1 );
		}
		if( layer->mDataTensorFactory == nullptr )
			layer->mDataTensorFactory = mDataFactory;
		if( layer->mWeigthTensorFactory == nullptr )
			layer->mWeigthTensorFactory = mWeightFactory;
		if ( !mLayers.empty() ) { //first layer needs to init output
			TensorP<ValueType> outputPrev = mLayers.back()->output();
			layer->input( outputPrev );
			Shape outputShape = layer->outputShape(); 
			std::cout << layer->name() << "   " << outputShape << std::endl;
			if ( !layer->buildsOwnOutputTensor() )
				layer->output( mDataFactory->create( outputShape ) );
		} else {
			Shape temp = layer->outputShape();
			std::cout << layer->name() << "   " << temp << std::endl;
		}

		// if the layer needs some extra setup
		layer->setup();

		mLayers.push_back( layer );

		if ( ( mUsage & MemoryUsage::greedy ) ) { // greedy memory allocation
			layer->initTensors();
			if ( ( mUsage & MemoryUsage::pre_convert_weights ) ){
				std::vector<TensorP<ConvertType>> c;
				for( auto tensor : layer->allWeights() )
					c.push_back( mWeightConverter->convert( tensor ) );
				layer->convertedWeights( c );
			}
		} else { //lazy TODO (get it?? because it is lazy??:

		}
	}


	void run() {
//		if ( !built ) {
//			std::cerr << "model not built!" << std::endl;
//			exit( 1 );
//		}
		auto start = std::chrono::system_clock::now();
		// Some computation here
		//ensure we have at least 1 layer
		assert( mLayers.size() > 0 );

		//traverse through layers and feeds forward
		for ( auto layer : mLayers ) {
			std::cout << layer->name() << std::flush;
			auto startLayer = std::chrono::system_clock::now();
			if ( (mUsage & MemoryUsage::pre_convert_weights) && (mUsage & MemoryUsage::lazy) ){
				std::vector<TensorP<ConvertType>> c;
				for( auto tensor : layer->allWeights() )
					c.push_back( mWeightConverter->convert( tensor ) );
				layer->convertedWeights( c );
			}
			layer->feedForward();
			auto endLayer = std::chrono::system_clock::now();
			std::chrono::duration<double> elapsed_seconds_layer = endLayer - startLayer;
			std::cout << " " << elapsed_seconds_layer.count() << "s" << std::endl;
			if( mUsage & MemoryUsage::free_after_use )
				layer->clear();
		}
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		std::cout << "Model run took: " << elapsed_seconds.count() << "s" << std::endl;
	}

	std::vector<int> classification() {
		if ( mLayers.back()->output()->shape.size != 2 )
			throw std::logic_error( "don't know how to do classification for this layer" );
		return mLayers.back()->output()->argmaxVector();
	}

	template<class LabelType>
	float accuracy( std::vector<LabelType> labels ) {
		return accuracy( classification(), labels );

	}

	template<class LabelType>
	float accuracy( std::vector<int> predictions, std::vector<LabelType> labels ) {
		int numCorrect = 0;
		for ( size_t i = 0; i < predictions.size(); ++i ) {
			if ( labels[ i ] == predictions[ i ] )
				numCorrect++;
		}
		return numCorrect / ( (float) predictions.size() );
	}

	void clearGraph() { // FIXME there should be a more elegant way
		for ( auto l : mLayers ) {
			l->input()->init();
			l->output()->init();
		}
	}

	template<class InType, class LabelType>
	float evaluate( std::vector<std::vector<InType>>& X,
			std::vector<LabelType>& Y ) {
		std::vector<int> predictions;
		uint instancesUsed = 0;
		while ( instancesUsed < X.size() ) {
			clearGraph();
			// setup the input tensor
			auto usedAndPadding = feedInputTensor( X, Y, instancesUsed );
			instancesUsed = usedAndPadding.first;
			uint padding = usedAndPadding.second;

			// run the network
			run();
			// get the predicitons
			std::vector<int> preds = classification();
//			std::cout << preds << std::endl; //FIXME compile error
			predictions.insert( predictions.end(), preds.begin(), preds.end() );
//			std::cout << predictions << std::endl; //FIXME error
			std::cout << "[";
			for ( uint var = 0; var < predictions.size(); ++var )
				std::cout << (unsigned) Y [ var ] << " ";
			std::cout << "]" << std::endl;
			std::cout << "\r" << instancesUsed << '/' << X.size() << std::flush;
			for ( uint i = 0; i < padding; ++i )
				predictions.pop_back();
		}
		float acc = accuracy( predictions, Y );
		std::cout << std::endl << "Accuracy: " << acc << std::endl;
		return acc;

	}

	template<class InType, class LabelType>
	std::pair<uint, uint> feedInputTensor( std::vector<std::vector<InType>>& X, std::vector<LabelType>& Y, uint instanceUsed = 0 ) {
		uint _instancesUsed = instanceUsed;
		TensorP<ValueType> inputTensor = mLayers.front()->input();
		inputTensor->init();
		int batchSize = inputTensor->shape [ 0 ];
		//batch vector
		Shape trueShape = inputTensor->shape; // @suppress("Invalid arguments")
		uint padding = 0;
		bool dealWithChannels = trueShape.size > 3 && trueShape[ 1 ] > 1;
		if ( _instancesUsed < X.size() ) {
			// load batch
			// first reshape input and flatten last dimension
			if ( dealWithChannels ) // we have to deal with channels
				inputTensor->reshape( Shape( { trueShape[ 0 ], trueShape[ 1 ], trueShape[ 2 ] * trueShape[ 3 ] } ) );
			else
				inputTensor->reshape( Shape( { trueShape[ 0 ], trueShape.capacity() / trueShape[ 0 ] } ) );
			uint i = 0;
			for ( unsigned int batchIdx = _instancesUsed; batchIdx < batchSize + _instancesUsed; ++batchIdx ) {
				uint j = 0;
				if ( batchIdx < X.size() ) {
					for ( auto px : X [ batchIdx ] ) {
						if ( dealWithChannels )
							( *inputTensor )[ { i, trueShape[ 1 ], j++ } ] = static_cast<ValueType>( px );
						else
							( *inputTensor )[ { i, j++ } ] = static_cast<ValueType>( px );
					}
					i++;
				} else { //pad the rest with 0 or whatever

					for ( uint px = 0; px < inputTensor->shape[ trueShape.size - 1 ]; ++px ) {
						if ( dealWithChannels )
							( *inputTensor )[ { i, trueShape[ 1 ], j++ } ] = inputTensor->empty();
						else
							( *inputTensor )[ { i, j++ } ] = inputTensor->empty();
					}
					padding++;
					i++;
				}
			}
			_instancesUsed += batchSize - padding;
			// resorte old shape
			inputTensor->reshape( trueShape );
//			std::cout << *inputTensor << std::endl;
		}
		return std::pair<uint, uint>( _instancesUsed, padding );
	}


	//model summary where we print out shapes, name of all layers
	void summary() {
		//ensure we have at least 1 layer
		assert( mLayers.size() > 0 );

		std::cout << "Model summary:" << std::endl;

		std::cout << "Layer\t\t" << "Output Shape" << std::endl;
		std::cout << "===========================================" << std::endl;
		for ( auto layer : mLayers ) {
			layer->description();
		}
		std::cout << "===========================================" << std::endl;
	}

	/**
	 * Loads the weights and also inits the weight tensors of each layer
	 */
	void loadWeights( std::string path, std::vector<std::string> names =
			std::vector<std::string>() ) {
		uint i = 0;
		std::thread tt[ mLayers.size() ];
		auto start = std::chrono::system_clock::now();
		for ( auto layer : mLayers ) {
			std::cout << "loading weights for" << layer->name() << std::endl;

			if ( names.empty() ) {
				tt[ i ] = std::thread( [=] {layer->loadWeights( path );} );
			} else {
				tt[ i ] = std::thread( [=] {layer->loadWeights( path, names[i] );} );
			}
			i++;
		}
		for ( uint j = 0; j < mLayers.size(); ++j )
			tt[ j ].join();
		auto end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = end - start;
		std::cout << "loading weights took: " << elapsed_seconds.count() << "s" << std::endl;

	}

	/**
	 * @brief inits all weights to a random value from [0,1]
	 */
	void randomWeights( ) {
		for ( auto layer : mLayers )
			layer->randomInit();
	}

	TensorP<ValueType> input(){
		return mLayers.front()->input();
	}

	TensorP<ValueType> output(){
		return mLayers.back()->output();
	}

	std::vector<LayerP<ValueType, WeightType, DataTensorType, WeightTensorType>> layers() const {
		return mLayers;
	}

	std::vector<LayerP<ValueType, WeightType, DataTensorType, WeightTensorType>> mLayers; //FIXME only here for debuggin
private:
	//need storage stuff here
	MemoryUsage mUsage; //used to maintain the type of memory usage we want
	bool built = false;
	TensorFactory<ValueType>* mDataFactory;
	TensorFactory<WeightType>* mWeightFactory;
	WeightConverter<WeightType,ConvertType>* mWeightConverter;
};



#endif /* ARCHITECTURE_MODELIMPL_H_ */
