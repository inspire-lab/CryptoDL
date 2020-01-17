/*
 * JSONModel.cpp
 *
 *  Created on: Apr 17, 2019
 *      Author: agustin
 */

#include "JSONModel.h"


//checks whether a layer is in current implementation of architecture
//TODO: global container and define in main to avoid reinitializing
bool isValidLayerClass( std::string layer ) {
	//container of all valid
	std::unordered_map<std::string, bool> validLayers;
	validLayers [ "Conv2D" ] = true;
	validLayers [ "Dense" ] = true;
	validLayers [ "Flatten" ] = true;
	validLayers [ "ZeroPadding2D" ] = true;
	validLayers [ "SimpleRNN" ] = true;

	//check if in container
	if ( validLayers.find( layer ) == validLayers.end() )
		return false;
	return true;
}

//template<class ValueType, class WeightType, class DataTensorType, class WeightTensorType>
//const std::string ModelLoader<ValueType, WeightType, DataTensorType, WeightTensorType>::python_code =
//"\"from kalypso import model_exporter \n"
//"import sys \n"
//"model_exporter( sys.argv[ 1 ], sys.argv[ 2 ], sys.argv[ 3 ] ) \"";
//
//template<class ValueType, class WeightType, class DataTensorType, class WeightTensorType>
//const std::string ModelLoader<ValueType, WeightType, DataTensorType, WeightTensorType>::command = "python -c " + python_code;
