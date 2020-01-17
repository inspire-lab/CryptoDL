/*
 * CompleteNetworkTests.h
 *
 *  Created on: Feb 5, 2019
 *      Author: robert
 */

#ifndef TEST_COMPLETENETWORKTESTS_H_

#define TEST_COMPLETENETWORKTESTS_H_

#include <gmpxx.h>
#include <limits>
#include <vector>
#include "../src/architecture/LoadModelData.h"
#include "../src/data/DatasetOperations.h"
#include "TestCommons.h"
#include "../src/architecture/ActivationFunction.h"
#include "../src/architecture/Layer.h"
#include "../src/architecture/PlainTensor.h"
#include "../src/architecture/Model.h"

inline std::vector<std::vector<std::vector<long>>> getTestBatch( std::vector<std::vector<uint8_t>> data ) {
	std::vector<std::vector<std::vector<long>>> plainImages = std::vector<std::vector<std::vector<long>>>( 32,
			std::vector<std::vector<long>>( 1, std::vector<long>() ) );
	for (int batchIdx = 0; batchIdx < 32; batchIdx++) {
		for( auto px : data[batchIdx] )
			plainImages[batchIdx][0].push_back(px);
	}
	return plainImages;
}

template<class T>
std::vector<std::vector<std::vector<T>>> loadConvLayerOutput1( std::string filename, int batchSize, int depth, int size,
		bool surpressOutput = false, std::string path = "src/keras/data/reference_output/" ) {
	std::string m_path = path + filename;
	std::ifstream ifs;
	ifs.open( m_path, std::ios::in );

	if ( ifs.fail() ) {
		std::cout << "Cannot find file " << m_path << std::endl;
		perror( "\nOpening input file error" );
		exit( 1 );
	}
	std::vector<std::vector<std::vector<T>>> out( batchSize,
			std::vector<std::vector<T>>( depth, std::vector<T>( size, 0 ) ) );
	T number;
	int batchIdx = 0;
	int depthIdx = 0;
	int pixelIdx = 0;
	int count = 0;
	while ( ifs >> number ) {
		if ( pixelIdx == size ) {
			pixelIdx = 0;
			depthIdx++;
		}
		if ( depthIdx == depth ) {
			depthIdx = 0;
			batchIdx++;
		}
		if ( batchIdx < batchSize ) {
			assert( batchIdx < batchSize && depthIdx < depth && pixelIdx < size );
			out [ batchIdx ] [ depthIdx ] [ pixelIdx ] = number;
			pixelIdx++;
		}
		count++;
	}
	if ( !surpressOutput )
		std::cout << "number read: " << count << " number expected: " << batchSize * depth * size << std::endl;
	return out;
}

template<class T>
std::vector<std::vector<T>> loadDenseLayerOutput1( std::string filename, int batchSize, int size,
		bool surpressOutput = false,
		std::string path = "src/keras/data/reference_output/" ) {
	std::string m_path = path + filename;
	std::ifstream ifs;
	ifs.open( m_path, std::ios::in );

	if ( ifs.fail() ) {
		std::cout << "Cannot find file " << m_path << std::endl;
		exit( 1 );
	}
	std::vector<std::vector<T>> out( batchSize, std::vector<T>( size, 0 ) );
	T number;
	int batchIdx = 0;
	int pixelIdx = 0;
	int count = 0;
	while ( ifs >> number ) {
		if ( pixelIdx == size ) {
			pixelIdx = 0;
			batchIdx++;
		}
		if ( batchIdx != batchSize ) {
			out [ batchIdx ] [ pixelIdx ] = number;
			pixelIdx++;
		}
		count++;
	}
	if ( !surpressOutput )
		std::cout << "number read: " << count << " number expected: " << batchSize * size << std::endl;
	return out;
}



bool completeNetworkTestLong();

bool completeNetworkTestFloat();

bool completeNetworkTestmpz();

bool completeNetworkTestHELibBFV();

bool completeNetworkTestHELibCKKS();

bool compareLayerByLayerEncryptedFloat();

bool completeNetworkTestFloatLegacy();

// all weights are 1
bool completeNetworkTestOneWeightsLong();

bool completeNetworkTestOneWeightsmpz();


// network with float weights
bool completeFloatNetworkTest();

// network with float weights and relu activation function
bool completeReluNetworkTest();

// network with float weights - cut off precision
bool completeFloatNetworkTest(int n);

bool compareLayerByLayerLong();

bool completeNetworkTestHELibCKKSFloatWeights();





#endif /* TEST_COMPLETENETWORKTESTS_H_ */
