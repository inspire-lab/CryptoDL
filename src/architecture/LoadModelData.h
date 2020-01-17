/*
 * LoadModelData.h
 *
 *  Created on: Jan 28, 2019
 *      Author: agustin
 */

#ifndef LOADMODELDATA_H_
#define LOADMODELDATA_H_

#include <fstream>
#include <vector>
#include <iomanip>
#include <sstream>
#include <type_traits>
#include <utility>

/*** WEIGHT LOADING CODE ***/


template<class T>
std::vector<T> loadFilterWeights( std::string name ) {
	std::ifstream FilterFile;
	float weight;
	std::vector<T> weights;
	FilterFile.open(name);

	if(FilterFile.fail()){
		std::cerr << "Cannot find file " << name << std::endl;
		}
	else{
		while(FilterFile >> weight){
			weights.push_back(weight);
		}
	}
	FilterFile.close();

	return weights;
}


template<class T>
std::vector<std::vector<T> > loadingFCWeights( std::string Address, int nomNeurons ) {
	std::vector<std::vector<T>> weightsVector;
	//Loading neurons weights for the first FC layer.
	for (int f = 0; f < nomNeurons; f++) {
		std::vector<T> weights;
		std::string s1 = std::to_string(f);
		std::string NeuronFileName = Address + "_" + s1 + ".txt";

		weights = loadFilterWeights<T>(NeuronFileName);
		weightsVector.push_back(weights);
	}
	return weightsVector;
}

template<class T>
std::pair<std::vector<std::vector<T>>, std::vector<T>> loadFilterWeights( std::string Address, int nomFilters,
		int nomFiltersPrevLayer, int filterSize ) {
	std::vector<std::vector<T> > filters;
	std::vector<T> bias;
	if ( nomFiltersPrevLayer == 1 ) {
		for ( int f = 0; f < nomFilters; f++ ) {
			std::vector<T> weights;
			std::string s = std::to_string( f );
			std::string FilterFileName = Address + "-0_" + s + ".txt";

			// std::cout << "loading " << FilterFileName << std::endl;

			weights = loadFilterWeights<T>( FilterFileName );
			bias.push_back( weights.back() );
			weights.pop_back();
			filters.push_back( weights );
		}
	}
	else {
		// if you think of the files building a matrix we need to extract the columns to get the filters that belong together
		for ( int f = 0; f < nomFilters; f++ ) {
			std::string s1 = std::to_string( f );
			for ( int g = 0; g < nomFiltersPrevLayer; g++ ) {
				std::string s2 = std::to_string( g );
				std::vector<T> weights;
				std::string FilterFileName = Address + "-" + s2 + "_" + s1 + ".txt"; // this is original

//				 std::cout << "loading " << FilterFileName << std::endl;

				weights = loadFilterWeights<T>( FilterFileName );
				if ( g == 0 ) // just once
					bias.push_back( weights.back() );
				weights.pop_back();
				filters.push_back( weights ); //grabs the last element at weights (bias) and adds to filter.
			}
		}
	}
	return std::pair<std::vector<std::vector<T>>, std::vector<T>>( filters, bias );
}

/// old
template<class T>
std::vector<std::vector<T> > loadingFilterWeights( std::string Address, int nomFilters, int nomFiltersPrevLayer,
		int filterSize ) {
	std::vector<std::vector<T> > filters;

	if (nomFiltersPrevLayer == 1) {
		for (int f = 0; f < nomFilters; f++) {
			std::vector<T> weights;
			std::string s = std::to_string(f);
			std::string FilterFileName = Address + "0_" + s + ".txt";

			weights = loadFilterWeights<T>(FilterFileName);
			filters.push_back(weights); //grabs the last element at weights (bias) and adds to filter. Filter will now contain nomFilters**2 + 1 elements
		}
	}
	else {
		for (int g = 0; g < nomFiltersPrevLayer; g++) {
			std::string s2 = std::to_string(g);
			for (int f = 0; f < nomFilters / nomFiltersPrevLayer; f++) {
				std::vector<T> weights;
				std::string s1 = std::to_string(f);
				std::string FilterFileName = Address + s2 + "_" + s1 + ".txt"; // this is original
				// The last element of returned weights has the bias value. I should separate it from the vector weights and save it in another variable.
				weights = loadFilterWeights<T>(FilterFileName);
				filters.push_back(weights); //grabs the last element at weights (bias) and adds to filter.
			}
		}
	}

	//filters is an array of arrays which contains nomFilters*nomFilters + 1 elements, with the last value being the bias
	return filters;
}

template<class T>
void getWeights( std::vector<std::vector<T>>& filtersConv1, std::vector<std::vector<T>>& filtersConv2,
		std::vector<std::vector<T>>& weightsVectorFC1, std::vector<T>& biasesFC1,
		std::vector<std::vector<T>>& weightsVectorFC2, std::vector<T>& biasesFC2,
		std::string Address = "savedWeights/" ) {
	/*** Start: Loading Weights ***/

	//	Loading weights for the first Conv layer
	std::string name = "conv2d_1-";
	int noOfFilters = 32;

	filtersConv1 = loadingFilterWeights<T>(Address + name, noOfFilters, 1, 5);

	//	Loading weights for the second Conv layer
	name = "conv2d_2-";
	noOfFilters = 2048;
	int noOfFiltersPrevLayer = 32;
	filtersConv2 = loadingFilterWeights<T>(Address + name, noOfFilters, noOfFiltersPrevLayer, 5);

	//	Loading weights for the first FC weights.
	name = "dense_1";
	weightsVectorFC1 = loadingFCWeights<T>(Address + name, 100);

	//Loading bias neuron for the first FC layer.
	name = Address + "dense_1_bias.txt";
	biasesFC1 = loadFilterWeights<T>(name);

	//Loading neurons weights for the second FC layer.
	name = "dense_2";
	weightsVectorFC2 = loadingFCWeights<T>(Address + name, 10);

	//Loading bias neuron for the second FC layer
	name = Address + "dense_2_bias.txt";
	biasesFC2 = loadFilterWeights<T>(name);
	/*** End: Loading Weights ***/
}


template<class T>
std::vector<T> loadFilterWeightsOfPrecision( std::string name, int precision ) {
	//ensure we are using at least one of these types
	assert( ( std::is_same<T, float>::value ) || ( std::is_same<T, double>::value ) );

	std::ifstream FilterFile;
	T weight;
	std::vector<T> weights;
	FilterFile.open(name);

	if(FilterFile.fail()){
		std::cerr << "Cannot find file " << name << std::endl;
		}
	else{
		while(FilterFile >> weight){
			//set precision
			std::stringstream ss;
			ss << std::fixed << std::setprecision( precision ) << weight;
			std::cerr << weight << " was converted into: " << std::stod( ss.str() ) << std::endl;
			//convert to T [TODO: potential error, does it to double, not float or anything else]
			if ( std::is_same<T, double>::value ) {
				weights.push_back(stod(ss.str()));
			}
			else if ( std::is_same<T, float>::value ) {
				weights.push_back(stof(ss.str()));
			}
		}
	}
	FilterFile.close();

	return weights;
}

template<class T>
std::vector<std::vector<T> > loadingFCWeightsOfPrecision( std::string Address, int nomNeurons, int precision ) {
	std::vector<std::vector<T>> weightsVector;
	//Loading neurons weights for the first FC layer.
	for (int f = 0; f < nomNeurons; f++) {
		std::vector<T> weights;
		std::string s1 = std::to_string(f);
		std::string NeuronFileName = Address + s1 + ".txt";

		weights = loadFilterWeightsOfPrecision<T>(NeuronFileName, precision);
		weightsVector.push_back(weights);
	}
	return weightsVector;
}

template<class T>
std::vector<std::vector<T> > loadingFilterWeightsOfPrecision( std::string Address, int nomFilters,
		int nomFiltersPrevLayer, int filterSize, int precision ) {
	std::vector<std::vector<T> > filters;

	if (nomFiltersPrevLayer == 1) {
		for (int f = 0; f < nomFilters; f++) {
			std::vector<T> weights;
			std::string s = std::to_string(f);
			std::string FilterFileName = Address + s + ".txt";

			weights = loadFilterWeightsOfPrecision<T>(FilterFileName, precision);
			filters.push_back(weights); //grabs the last element at weights (bias) and adds to filter. Filter will now contain nomFilters**2 + 1 elements
		}
	}
	else {
		for (int g = 0; g < nomFiltersPrevLayer; g++) {
			for (int f = 0; f < nomFilters / nomFiltersPrevLayer; f++) {
				std::vector<T> weights;
				std::string s1 = std::to_string(f);
				std::string s2 = std::to_string(g);
				std::string FilterFileName = Address + s2 + "_" + s1 + ".txt";
				//I only keep filter weights. The last element of returned weights has the bias value. I should separate it from the vector weights and save it in another variable.
				weights = loadFilterWeightsOfPrecision<T>(FilterFileName, precision);
				filters.push_back(weights); //grabs the last element at weights (bias) and adds to filter. Filter will now contain nomFilters**2 + 1 elements
			}
		}
	}

	//filters is an array of arrays which contains nomFilters*nomFilters + 1 elements, with the last value being the bias
	return filters;
}

template<class T>
void getWeightsOfPrecision( std::vector<std::vector<T>>& filtersConv1, std::vector<std::vector<T>>& filtersConv2,
		std::vector<std::vector<T>>& weightsVectorFC1, std::vector<T>& biasesFC1,
		std::vector<std::vector<T>>& weightsVectorFC2, std::vector<T>& biasesFC2, int precision, std::string Address =
				"savedWeights/" ) {
	/*** Start: Loading Weights ***/

	//	Loading weights for the first Conv layer
	std::string name = "conv2d_1-0_";
	int noOfFilters = 32;

	filtersConv1 = loadingFilterWeightsOfPrecision<T>(Address + name, noOfFilters, 1, 5, precision);

	//	Loading weights for the second Conv layer
	name = "conv2d_2-";
	noOfFilters = 2048;
	int noOfFiltersPrevLayer = 32;
	filtersConv2 = loadingFilterWeightsOfPrecision<T>(Address + name, noOfFilters, noOfFiltersPrevLayer, 5, precision);

	//	Loading weights for the first FC weights.
	name = "dense_1_";
	weightsVectorFC1 = loadingFCWeights<T>(Address + name, 100);

	//Loading bias neuron for the first FC layer.
	name = Address + "dense_1_bias.txt";
	biasesFC1 = loadFilterWeightsOfPrecision<T>(name, precision);

	//Loading neurons weights for the second FC layer.
	name = "dense_2_";
	weightsVectorFC2 = loadingFCWeightsOfPrecision<T>(Address + name, 10, precision);

	//Loading bias neuron for the second FC layer
	name = Address + "dense_2_bias.txt";
	biasesFC2 = loadFilterWeightsOfPrecision<T>(name, precision);
	/*** End: Loading Weights ***/
}



#endif /* LOADMODELDATA_H_ */
