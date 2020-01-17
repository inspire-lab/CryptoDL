/*
 * DataReaders.h
 *
 *  Created on: May 30, 2019
 *      Author: robert
 */

#ifndef TOOLS_DATAREADERS_H_
#define TOOLS_DATAREADERS_H_


#include "../architecture/Tensor.h"

/**
 * @brief Reads a file as 32 bit float numbers. It is not very portable.
 * Fails hard if floats are not 32bit on the system.
 */
std::vector<float> readFloat32FromBinary(const std::string& file );


/**
 * @brief Reads a file as 32 bit int numbers. It is not very portable.
 * Fails hard if it cant find int32_t on the system.
 */
std::vector<int32_t> readInt32FromBinary(const std::string& file );

std::vector<int8_t> readInt8FromBinary(const std::string& file );

/**
 * @brief Convert a vector of floats to doubles
 */
std::vector<double> floatToDouble(const std::vector<float>& floats );


/**
 * @brief Read png image and qunatized the data by dividing every pixel by the max value. Default maxvalue is 255.
 */

typedef std::vector<std::vector<std::vector<float>>> float_img_vec;
float_img_vec readJPGquantized( const std::string& file, float maxvalue=255. );

typedef std::vector<float> float_flat_img;
float_flat_img readJPGquantizedFlat( const std::string& file, float maxvalue=255. );




/**
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
template<typename T>
TensorP<T> readTensorFromBinary( const std::string& file );

#endif /* TOOLS_DATAREADERS_H_ */
