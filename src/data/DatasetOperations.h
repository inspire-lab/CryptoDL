/*
 * DatasetOperations.h
 *
 *  Created on: Jan 28, 2019
 *      Author: agustin
 */

#ifndef DATASETOPERATIONS_H_
#define DATASETOPERATIONS_H_

#include "mnist/include/mnist/mnist_reader_less.hpp"
#include "../tools/DataReaders.h"
#include "../tools/Config.h"


/*** DATASET OPERATION ***/

mnist::MNIST_dataset<uint8_t, uint8_t> loadMNIST( const std::string& path );

mnist::MNIST_dataset<uint8_t, uint8_t> loadMNIST();



/**
 * @brief loads the test data of the cowc dataset
 */
std::pair<std::vector<float_flat_img>, std::vector<u_int8_t>> loadCOWC( const std::string& path, float quantFactor=1. );

inline std::pair<std::vector<float_flat_img>, std::vector<u_int8_t>> loadCOWC( float quantFactor=1. ){
	return loadCOWC( Config::getConfig()->get<std::string>( "datasets", "cowc-home" ), quantFactor );
}


#endif /* DATASETOPERATIONS_H_ */


