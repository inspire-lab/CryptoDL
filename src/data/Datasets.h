/*
 * Datasets.h
 *
 *  Created on: May 1, 2019
 *      Author: agustin
 */

#ifndef DATASETS_H_
#define DATASETS_H_

#include "architecture/Tensor.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <stdexcept>
#include <boost/filesystem.hpp>//used to traverse through all subdirs for the imdb dataset

//Used to avoid long lines
typedef std::vector<unsigned int> row;
typedef std::vector<row> image;
typedef std::vector<image> image_channel;
typedef std::vector<image_channel> image_collection;

#define MAX_LEN 80

//necessary to grab the actual number from the binary files
//TODO: test
inline int reverseInt( int i ) {
	return ( (int) ( i & 255 ) << 24 ) + ( (int) ( ( i >> 8 ) & 255 ) << 16 )
			+ ( (int) ( ( i >> 16 ) & 255 ) << 8 ) + ( ( i >> 24 ) & 255 );
}

//Used dariush's https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
//implementation to load the mnist dataset into tensors of the correct size.
//Allows for the files to be wherever in the system and still load them in (unlike the previous
//mnist reader implm usage).

//TODO: probably make these member functions rather than regular functions
//https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
image_collection read_mnist_images( std::string full_path,
		unsigned int& number_of_images, unsigned int& image_size ) {
	//open binary data file
	std::ifstream file( full_path, std::ios::binary );

	if ( file.is_open() ) {
		int magic_number = 0, n_rows = 0, n_cols = 0;

		//read header
		file.read( (char *) &magic_number, sizeof ( magic_number ) );
		magic_number = reverseInt( magic_number );

		if ( magic_number != 2051 )
			throw std::runtime_error( "Invalid MNIST image file!" );

		//grab necessary information from header
		file.read( (char *) &number_of_images, sizeof ( number_of_images ) ), number_of_images =
				reverseInt( number_of_images );
		file.read( (char *) &n_rows, sizeof ( n_rows ) ), n_rows = reverseInt(
				n_rows );
		file.read( (char *) &n_cols, sizeof ( n_cols ) ), n_cols = reverseInt(
				n_cols );

		//calculate what the file says it should b e
		image_size = n_rows * n_cols;
		assert( n_rows == 28 && n_cols == 28 );

		//define a vector of number_of_images x 1 x 28 x 28, as there are number_of_images images; 1 channel (greyscale), 28 rows; 28 cols
		image_collection dataset( number_of_images,
				std::vector<std::vector<std::vector<unsigned int>>>( 1,
						std::vector<std::vector<unsigned int>>( 28,
								std::vector<unsigned int>( 28, 0 ) ) ) );

		//grab all images and place into container
		for ( size_t i = 0; i < number_of_images; i++ ) {
			//TODO: ensure this is correct, seems wrong
			unsigned char* _image = new unsigned char [ image_size ];
			file.read( (char *) _image, image_size );

			//copy over img to vector and cast to unsigned int
			for ( size_t j = 0; j < image_size; j++ ) {
				//calculate row and col indices
				size_t row = j / 28;
				size_t col = j % 28;

				dataset [ i ] [ 0 ] [ row ] [ col ] =
						(unsigned int) _image [ j ];
			}

			//free memory
			delete [ ] _image;
			_image = nullptr;
		}

		//close file
		file.close();

		return dataset;
	} else {
		throw runtime_error( "Cannot open file `" + full_path + "`!" );
	}
}

//https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
typedef unsigned char uchar;
std::vector<unsigned int> read_mnist_labels( std::string full_path,
		int& number_of_labels ) {
	std::ifstream file( full_path, std::ios::binary );

	if ( file.is_open() ) {
		int magic_number = 0;
		file.read( (char *) &magic_number, sizeof ( magic_number ) );
		magic_number = reverseInt( magic_number );

		if ( magic_number != 2049 )
			throw std::runtime_error( "Invalid MNIST label file!" );

		file.read( (char *) &number_of_labels, sizeof ( number_of_labels ) ), number_of_labels =
				reverseInt( number_of_labels );

		//read in labels into vector and cast to unsigned int
		uchar* _dataset = new uchar [ number_of_labels ];
		std::vector<unsigned int> dataset( number_of_labels );
		for ( int i = 0; i < number_of_labels; i++ ) {
			file.read( (char*) &_dataset [ i ], 1 );

			dataset [ i ] = (unsigned int) _dataset [ i ];
		}

		//close file
		file.close();

		//free memory
		delete [ ] _dataset;
		_dataset = nullptr;

		return dataset;
	} else {
		throw runtime_error( "Unable to open file `" + full_path + "`!" );
	}
}


template<class T>
TensorP<T> grabData_MNIST( std::string path, std::string type ) {
	TensorP<T> data;

	//read data in
	try {
		std::string fullFile;
		size_t size;

		//TODO: get platform agnostic method of concatenating path
		if ( type == "train" ) {
			//training set contains 60000 images, each image is 28x28
			fullFile = path + "/" + "train-images-idx3-ubyte";
			size = 60000;
		} else if ( type == "test" ) {
			//test set contains 10000 images, each image is 28x28
			fullFile = path + "/" + "t10k-images-idx3-ubyte";
			size = 10000;
		} else {
			throw runtime_error(
					"The MNIST dataset does not have " + type + "!" );
		}

		//set tensor size
		data = TensorP<T>( Shape { size, 1, 28, 28 } ); // @suppress("Symbol is not resolved")

		unsigned int numOfImages = size, imageSize = 28 * 28;
		//TODO: read pathname of data-set directory from a configuration file
		image_collection mData = read_mnist_images( fullFile, numOfImages,
				imageSize );

		//initialize tensor with the data
		data->init( mData );
	} catch ( const std::runtime_error& r_e ) {
		std::cerr << r_e.what() << std::endl;
		std::cerr << "Will return uninitialized Tensor" << std::endl;
	} catch ( ... ) {
		std::cerr << "Will return uninitialized Tensor" << std::endl;
	}

	return data;
}


//since the imdb dataset is a collection of files with their filenames acting as labels, we need
//to grab the filename and find the id and the rating assigned to it
//the id is used to ensure that the files are 'placed' in the same numerical order they are in their files
//has file convenction [[id]_[rating].txt] where [id] is a unique id and [rating] is the star rating for that review on a 1-10 scale.
std::pair<int, int> get_imdb_id_score( std::string filename ) {
	std::pair<int, int> info;

	//remove the .txt portion of filename as it is unnecessary
	std::size_t txt = filename.find( ".txt" );
	filename = filename.substr( 0, txt );

	size_t findUnder = filename.find( "_" );
	if ( findUnder == std::string::npos ) {
		//TODO: throw exception that the database is wrong
	} else {
		//split filename into id and rating
		std::string id = filename.substr( 0, findUnder );
		std::string rating = filename.substr( findUnder + 1 );

		info.first = stoi( id );
		info.second = stoi( rating );
	}

	return info;
}

/*Requires the boost/filesystem library since the imdb dataset is a bunch of files in 4 separate directories
 *We traverse through the pos subdirectory first then go into the neg subdirectory
 *Returns a pair of vectors of sentiments and ratings
 * */
std::pair<std::vector<std::string>, std::vector<unsigned int>> read_imdb_sentiments(
		std::string path ) {
	//create container housing 12500 strings
	std::vector<std::string> sentiments( 25000 );
	std::vector<unsigned int> ratings( 25000 );

	//starts in the pos sub dir
	std::string currentPath = path + "pos/";

	//local data structure to hold the id and rating of a specific file
	std::pair<int, int> idScore;
	std::string line;

	//used to hold both sentiments and ratings
	std::pair<std::vector<std::string>, std::vector<unsigned int>> data;

	//loop over the neg and pos sub-directories
	for ( size_t i = 0; i < 2; i++ ) {
		try {
			//find offset for inserting into vectors
			int offset = i * 12500;

			//iterate over the directory in the current path and go through each file within it
			for ( auto j = boost::filesystem::directory_iterator( currentPath );
					j != boost::filesystem::directory_iterator(); j++ ) {
				if ( !is_directory( j->path() ) ) //we eliminate directories
						{
					//grab complete file path and name
					std::string filename(
							currentPath + j->path().filename().string() );

					//open file
					std::ifstream file( filename );

					//read in the line and get the first MAX_LEN characters of it
					getline( file, line );
					line = line.substr( 0, MAX_LEN );

					//grab id and rating
					idScore = get_imdb_id_score(
							j->path().filename().string() );

					//store the sentiment and score
					sentiments [ idScore.first + offset ] = line;
					ratings [ idScore.first + offset ] = idScore.second;
				} else
					continue; //is a directory, do not want to use (should not happen)
			}

		} catch ( ... ) {
			std::cerr << "Have more careful exception handling" << std::endl; //TODO!
		}

		//set the currentPath to the neg subdir for the next iteration
		currentPath = path + "neg/";
	}

	//return information
	data.first = sentiments;
	data.second = ratings;
	return data;
}

//gives us: test and train dir, each of which comes with neg and post subdirs
//	test/neg has 12500
//	test/pos has 12500
//	train/neg has 12500
//	train/pos has 12500
template<class T>
TensorP<T> grabDataIMDB( std::string path, std::string type ) {
	//contains 50,000 reviews
	//split into 25k train and 25k test sets

	//have additional 50k unlabeled reviews
	TensorP<T> data;

	//read data in
	try {
		std::string fullFile;
		size_t size;

		//TODO: get platform agnostic method of concatenating path
		if ( type == "train" ) {
			//training set contains 60000 images, each image is 28x28
			fullFile = path + "/" + "train/";
			size = 25000;
		} else if ( type == "test" ) {
			//test set contains 10000 images, each image is 28x28
			fullFile = path + "/" + "test/";
			size = 25000;
		} else {
			throw runtime_error(
					"The IMDB dataset does not have " + type + "!" );
		}

		//set tensor size
		//TODO: figure out dimensionality
		//data = TensorP<T>( Shape { size, 1, 28, 28 } );

		//TODO: read pathname of data-set directory from a configuration file
		//void mData = read_imdb_sentiments( fullFile );

		//initialize tensor with the data
		//data->init( mData );
	} catch ( const std::runtime_error& r_e ) {
		std::cerr << r_e.what() << std::endl;
		std::cerr << "Will return uninitialized Tensor" << std::endl;
	} catch ( ... ) {
		std::cerr << "Will return uninitialized Tensor" << std::endl;
	}
}

//A data structure that houses the training and testing datasets and labels
template<class DataType, class LabelType>
class Data {
private:
	TensorP<DataType> mTrainingSet;
	TensorP<DataType> mTestSet;
	TensorP<LabelType> mTrainingLabels;
	TensorP<LabelType> mTestLabels;
	Data();	//to avoid use
public:
	Data( std::string set );

	TensorP<DataType> grabData( std::string type );
	TensorP<LabelType> grabLabels( std::string type );

	//TODO: use memory.h's deleter method for smart pointers to have
	//'lazy' usage of dataset

};


#endif /* DATASETS_H_ */
