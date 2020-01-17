/*
 * DataReaders.cpp
 *
 *  Created on: May 30, 2019
 *      Author: robert
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <bits/stdc++.h>
#include <iostream>
#include <boost/gil/extension/io/jpeg_io.hpp>
#include <boost/algorithm/string.hpp>
#include "DataReaders.h"
#include "../architecture/PlainTensor.h"
namespace gil = boost::gil;



std::vector<float> readFloat32FromBinary(const std::string& file ) {
	if ( CHAR_BIT * sizeof( float ) != 32 ){
		std::cerr << " Error floats are not 32 bit on this platform. We are dealing with  " << CHAR_BIT * sizeof( float ) << "floats" << std::endl;
		exit( 1 );
	}

	std::ifstream fin( file, std::ios::binary );
	if ( !fin ) {
		std::cerr << " Error, Couldn't find " << file << std::endl;
		exit( 1 );
	}

	fin.seekg( 0, std::ios::end );
	const size_t num_elements = fin.tellg() / sizeof( float ) ;
	fin.seekg( 0, std::ios::beg );

	std::vector<float> data( num_elements );
	fin.read( reinterpret_cast<char*>( &data[ 0 ] ), num_elements * sizeof( float ) );
	return data;
}

std::vector<int32_t> readInt32FromBinary(const std::string& file ) {
	if ( CHAR_BIT * sizeof( int32_t ) != 32 ){
		std::cerr << " Error! Can not find 32 bit int on this platform. We are dealing with  " << CHAR_BIT * sizeof( int32_t ) << "int" << std::endl;
		exit( 1 );
	}

	std::ifstream fin( file, std::ios::binary );
	if ( !fin ) {
		std::cerr << " Error, Couldn't find " << file << std::endl;
		exit( 1 );
	}

	fin.seekg( 0, std::ios::end );
	const size_t num_elements = fin.tellg() / sizeof( int32_t ) ;
	fin.seekg( 0, std::ios::beg );

	std::vector<int32_t> data( num_elements );
	fin.read( reinterpret_cast<char*>( &data[ 0 ] ), num_elements * sizeof( int32_t ) );
	return data;
}

std::vector<int8_t> readInt8FromBinary(const std::string& file ){
	if ( CHAR_BIT * sizeof( int8_t ) != 8 ){
		std::cerr << " Error! Can not find 8 bit int on this platform. We are dealing with  " << CHAR_BIT * sizeof( int8_t ) << "int" << std::endl;
		exit( 1 );
	}

	std::ifstream fin( file, std::ios::binary );
	if ( !fin ) {
		std::cerr << " Error, Couldn't find " << file << std::endl;
		exit( 1 );
	}

	fin.seekg( 0, std::ios::end );
	const size_t num_elements = fin.tellg() / sizeof( int8_t ) ;
	fin.seekg( 0, std::ios::beg );

	std::vector<int8_t> data( num_elements );
	fin.read( reinterpret_cast<char*>( &data[ 0 ] ), num_elements * sizeof( int8_t ) );
	return data;
}



std::vector<double> floatToDouble(const std::vector<float>& floats ){
	std::vector<double> doubleVec( floats.begin(), floats.end() );
	return doubleVec;
}



float_img_vec readJPGquantized( const std::string& file, float maxvalue ){

	 gil::rgb8_image_t img;
	 gil::jpeg_read_image( file , img );
	 auto w = img.width();
	 auto h = img.height();
	 std::cout << "Read complete, got an image " << w << " by " << h << " pixels" << std::endl;
	 // build output vector
	 float_img_vec imageVector;
	 std::cout << "reserving space";
	 imageVector.resize( 3 ); // 3 channels
	 int count = 0;
	 std::cout << ".";
	 for( auto& channel : imageVector ){
		 channel.resize( h ); // reserve h rows
	         std::cout << ".";
		 for( auto& row : channel ){
			 row.resize( w ); // resrve for we pixels
			 std::cout << ".";
	         }
	 }
         std::cout << "done " << std::endl;

	 for( uint y = 0; y < h; ++y ){
		 for( uint x = 0; x < w; ++x ){
			 gil::rgb8_pixel_t px = *const_view(img).at(x, y);
			 // std::cout << "read pixel\n ";
			 // std::cout << imageVector.size() << std::endl ;
			 // std::cout << imageVector[0].size() << std::endl;
			 // std::cout << imageVector[0][y].size() << std::endl;
			 imageVector[0][y][x] = px[0] / maxvalue;
			 imageVector[1][y][x] = px[1] / maxvalue;
			 imageVector[2][y][x] = px[2] / maxvalue;
			 // std::cout << x << ',' << y << std::endl;
		 }
	 }

	return imageVector;
}

std::vector<float> readJPGquantizedFlat( const std::string& file, float maxvalue ){
	gil::rgb8_image_t img;
	gil::jpeg_read_image( file , img );
	auto w = img.width();
	auto h = img.height();
//	std::cout << "Read complete, got an image " << img.width() << " by " << img.height() << " pixels" << std::endl;
	// build output vector
	std::vector<float> imageVector;
	imageVector.resize( 3 * w * h ); // 3 channels

	int i = 0;
	for( uint c = 0; c < 3; ++c ){
		for( uint y = 0; y < h; ++y ){
			for( uint x = 0; x < w; ++x ){
				gil::rgb8_pixel_t px = *const_view(img).at(x, y);
				imageVector[i++] = px[c] / maxvalue;
			}
		}
	}
	return imageVector;
}





template<>
TensorP<float> readTensorFromBinary( const std::string& file ){
	/*
	 *     binary file format:

    version 0:
        [ 1byte: version ][ 1byte: reserved ][ 4byte: no dimension n ][ n*4 byte: dimensions ][ 1byte: datatype ][ ...data... ]
    datatypes:
        version: unsigned char
        no dimensions n: unsigned char
        dimensions: 32bit unsignet integer
        datatype: unsigned char
    meaning of the fields:
        version:
            what version of the file is being read headerfields might change depending on the version.
            current version: 0
        reserved:
            currently unused
        no dimensions:
            unsigned integer that indicates the number of dimesions of the stored tensor
        dimensions:
            a number of unsigned ints that indicate the number of elements for each tensor dimension. the number of unsigned ints
            is given by the previous field
        datatype:
            the datatype of the tensor. supported types:
            0: int32
            1: float32
        data:
            the actual payload. the size should be the product of all dimensions * size_of( dataype ) in bytes

	 */


	// open file
	std::ifstream fin( file, std::ios::binary );
	if ( !fin ) {
		std::cerr << " Error, Couldn't find " << file << std::endl;
		exit( 1 );
	}

	// read version
	unsigned char version;
	fin.read( reinterpret_cast<char*>( &version ), sizeof( version ) );
	if ( version != 0 ){
		// only support version 0 so far
		std::cerr << " Unsupported version " << (unsigned) version << std::endl;
		exit( 1 );
	}

	// skip reserved byte
	fin.ignore( 1 );

	// read number of dimensions
	uint32_t dims;
	fin.read( reinterpret_cast<char*>( &dims ), sizeof( dims ) );

	// read the shape as a vector
	std::vector<uint32_t> shape_v( dims );
	fin.read( reinterpret_cast<char*>( &shape_v ), dims * sizeof( uint32_t ) );

	// read the type info
	unsigned char datatype;
	fin.read( reinterpret_cast<char*>( &datatype ), sizeof( datatype ) );

	// only floats in this function
	if( datatype != 1 ){
		// only support version 0 so far
		std::cerr << " Unsupported datatype " << (unsigned) version << std::endl;
		exit( 1 );
	}

	// read the rest of the data
	Shape shape( shape_v );
	std::vector<float> data_v( shape.size );
	fin.read( reinterpret_cast<char*>( &data_v ), shape.size * sizeof( float ) );

	PlainTensorFactory<float> tf;

	auto tensor = tf.create(shape);
	tensor->flatten();
	tensor->init( data_v );
	tensor->reshape( shape );

	return tensor;

}













































































