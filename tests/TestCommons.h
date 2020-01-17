/*
 * TestCommons.h
 *
 *  Created on: Jan 30, 2019
 *      Author: robert
 */

#ifndef TEST_TESTCOMMONS_H_

#define TEST_TESTCOMMONS_H_

#include <fstream>
#include <iomanip>
#include <vector>
#include <stack>
#include <string>
#include <algorithm>
#include <type_traits>
#include <cmath>
#include <cassert>
#include "../src/architecture/Tensor.h"
#include "../src/architecture/PlainTensor.h"


template<class T, class U>
bool finishTest( TensorP<T> output, TensorP<U> expectedOutput, std::string testName ) {
	// bounds check
	bool passed = true;
	if ( output->shape != expectedOutput->shape ) {
		passed = false;
		std::cout << "output shape " << output->shape << std::endl;
		std::cout << "expected shape " << expectedOutput->shape << std::endl;
	}
	Shape orgShape = output->shape;
	output->flatten();
	expectedOutput->flatten();
	T sum = 0;
	int count = 0;
	for ( uint i = 0; i < output->shape.capacity(); ++i ) {
		if ( ( *output )[ { i } ] != ( *expectedOutput )[ { i } ] ) {
			passed = false;
			sum += std::abs( ( *output ) [ { i } ] - ( *expectedOutput ) [ { i } ] );
			if ( count++ < 32 )
				std::cout << std::fixed << std::setprecision(9) << i << " " << ( *output )[ { i } ] << " " << ( *expectedOutput )[ { i } ] << std::endl;
		}
	}
	sum /= (float) orgShape.capacity();
	output->reshape( orgShape );
	if ( passed ) {
		std::cout << "Passed " << std::endl;
		return true;
	}
//	std::cout << "expected Output:" << std::endl;
//	std::cout << *expectedOutput << std::endl;
//	std::cout << "our Output:" << std::endl;
//	std::cout << *output << std::endl;
	errno = -1;
	std::cout << testName << " average error: " << sum << std::endl;
	std::string str = "Failed " + testName;
	perror( str.c_str() );
	return false;

}


template<class T>
void printResult( bool passed, T ourResult, T expectedOutput, std::string testName ){
	if(passed)
		std::cout << "Passed " << testName << std::endl;
	else{
		std::cout << "expected Output:" << std::endl;
		std::cout << expectedOutput << std::endl;
		std::cout << "our Output:" << std::endl;
		std::cout << ourResult << std::endl;
		errno = -1;
		std::string str = "Failed " + testName;
		perror(str.c_str());
	}
}

template<class T>
bool compareOuput( std::vector<std::vector<std::vector<T>>> one, std::vector<std::vector<std::vector<T>>> two ){
	if( one.size() != two.size() || one[0].size() != two[0].size() || one[0][0].size() != two[0][0].size()  )
		return false;
	for (unsigned int i = 0; i < one.size(); ++i)
		for (unsigned int j = 0; j < one[0].size(); ++j)
			for (unsigned int k = 0; k < one[0][0].size(); ++k)
				if( one[i][j][k] != two[i][j][k] )
					return false;
	return true;
};

template<class T>
bool compareOuput( std::vector<std::vector<T>> one, std::vector<std::vector<T>> two ){
	if( one.size() != two.size() || one[0].size() != two[0].size()   )
		return false;
	for (unsigned int i = 0; i < one.size(); ++i)
		for (unsigned int j = 0; j < one[0].size(); ++j)
			if( one[i][j] != two[i][j] )
				return false;
	return true;
};


template<class T>
bool finishTest(std::vector<std::vector<std::vector<T>>> ourResult, std::vector<std::vector<std::vector<T>>> expectedOutput, std::string testName){
	bool result =  compareOuput(ourResult, expectedOutput);
	printResult(result, ourResult, expectedOutput, testName);
	return result;
}

template<class T>
bool finishTest(std::vector<std::vector<T>> ourResult, std::vector<std::vector<T>> expectedOutput, std::string testName){
	bool result =  compareOuput(ourResult, expectedOutput);
	printResult(result, ourResult, expectedOutput, testName);
	return result;
}

template<class T>
std::vector<std::vector<std::vector<T>>> loadFromDataset(int nImages, int size, std::string filename, bool surpresOutput=false){
  std::ifstream input_file(filename);
  if(!input_file.is_open()){
	  std::cout << filename << std::endl;
	  perror("\nOpening input file error");
	  exit(1);
  }
  std::vector<std::vector<std::vector<T>>> out( nImages, std::vector<std::vector<T>>{ std::vector<T>( size, 0 )  } );
  T t;
  int batchIdx = 0;
  int pixelIdx = 0;

  while(input_file >> t){
    if( pixelIdx == size ){
    	pixelIdx = 0;
    	batchIdx ++;
    }
    if( batchIdx == nImages )
    	break;
    out[batchIdx][0][pixelIdx++] = t;
  }
  return out;
}


template<class T>
std::vector<T> loadLayerOutput( std::string filename, int readNumber = -1, bool surpressOutput = false, std::string path = "src/keras/data/reference_output/" ) {
	std::string m_path = path + filename;
	std::ifstream ifs;
	ifs.open( m_path, std::ios::in );

	if ( ifs.fail() ) {
		std::cout << "Cannot find file " << m_path << std::endl;
		exit( 1 );
	}
	std::vector<T> out;
	T number;
	int count = 0;
	while ( ifs >> number ) {
		out.push_back( number );
		count++;
		if ( readNumber != -1 && readNumber == count )
			break;
	}
	if ( !surpressOutput )
		std::cout << "number read: " << count << std::endl;
	return out;
}

////TODO: can try to determine tensor shape ourselves but can skip it for testing
////took the easy way out and just have it ask user for shape
template<class T>
TensorP<T> createTensorFromKeras( std::vector<unsigned int> shape,
		std::string kerasOutput ) {
	assert( shape.size() == 4 );

	std::vector<double> container; //assume float for stod then we can static cast to correct type
	std::string curValue = ""; //used to maintain an entire value at a time

	//sanitize string
	for ( unsigned int i = 0; i < kerasOutput.size(); i++ ) {
		if ( !isdigit( kerasOutput [ i ] ) ) {
			if ( kerasOutput [ i ] != '.' )
				kerasOutput [ i ] = ' ';
		}
	}

	//create container of elements from string
	for ( unsigned int i = 1; i < kerasOutput.size(); i++ ) {
		if ( kerasOutput [ i ] == '.' || isdigit( kerasOutput [ i ] ) ) {
			curValue += std::string( 1, kerasOutput [ i ] );
		}
		else {
			if ( kerasOutput [ i - 1 ] == '.'
					|| isdigit( kerasOutput [ i - 1 ] ) ) {
				try {
					container.push_back( stod( curValue ) );
				} catch ( const std::invalid_argument & ia ) {
					std::cerr << "Invalid argument: " << ia.what() << std::endl;
					exit( EXIT_FAILURE );
				}
			}

			curValue = "";
		}
	}

	//cast to correct container
	std::vector<T> castContainer;
	if ( std::is_same<T, double>::value ) {
		for ( double el : container ) { //is stupid but allows compilation
			castContainer.push_back( (double) el );
		}
	} else if ( std::is_same<T, float>::value ) {
		for ( double el : container ) {
			castContainer.push_back( (float) el );
		}
	} else if ( std::is_same<T, int>::value ) {
		for ( double el : container ) {
			castContainer.push_back( (int) el );
		}
	} else if ( std::is_same<T, long>::value ) {
		for ( double el : container ) {
			castContainer.push_back( (long) el );
		}
	} else {
		std::cerr << "Invalid template." << std::endl;
	}

	PlainTensorFactory<T> dataFactory;

	//create a 1d tensor that can holds a*b*c*d values
	unsigned int size = shape [ 0 ] * shape [ 1 ] * shape [ 2 ] * shape [ 3 ];
	TensorP<T> kOut = dataFactory.create( Shape( { size } ) );

	kOut->init( castContainer );
	kOut->reshape( { shape [ 0 ], shape [ 1 ], shape [ 2 ], shape [ 3 ] } );

	return kOut;

	//code for having program find dimensions, needs logic
	//traverse through string
	//std::stack<std::string> elements;
	/*for ( unsigned int i = 0; i < kerasOutput.size(); i++ ) {
		std::cout << std::endl << "Current char: " << kerasOutput [ i ] << "-.-"
				<< std::endl;

		if ( kerasOutput [ i ] == '[' ) { //open bracket, we have some sort of container
			elements.push( std::string( 1, kerasOutput [ i ] ) );
		} else if ( kerasOutput [ i ] == ']' ) { //end bracket, we have ended the last open container
			if ( isdigit( kerasOutput [ i - 1 ] )
					|| kerasOutput [ i - 1 ] == '.' ) {
				elements.push( curValue );
				curValue = "";
			}

			std::vector<std::string> row;

			//pop the entire 'row' of elements
			while ( elements.top() != "[" ) {
				row.push_back( elements.top() );
				elements.pop();
			}
			elements.top(); //remove the [

			if ( !row.empty() ) {
				std::reverse( row.begin(), row.end() ); //reverse the row to be in correct order

				std::cout << "Row contents: " << std::endl;
				for ( std::string elem : row ) {
					std::cout << elem << std::endl;
				}

				//place all elements into our container
				for ( std::string elem : row ) {
					try {
						std::cout << "-" << elem << "- ";
						container.push_back( stod( elem ) );
					} catch ( const std::invalid_argument & ia ) {
						std::cerr << "Invalid argument: " << ia.what()
								<< std::endl;
						exit( EXIT_FAILURE );
					}
				}
			}

		} else if ( kerasOutput [ i ] == ' ' ) { //delimiting char, place value into container
			elements.push( curValue );
			curValue = ""; //reset cur value
			continue;
		} else if ( isdigit( kerasOutput [ i ] ) || kerasOutput [ i ] == '.' ) { //have a number
			curValue += std::string( 1, kerasOutput [ i ] );
		}
	 }*/
}



#endif /* TEST_TESTCOMMONS_H_ */
