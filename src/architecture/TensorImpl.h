/*
 * TensorImpl.h
 *
 *  Created on: Feb 18, 2019
 *      Author: robert
 */

#ifndef ARCHITECTURE_TENSORIMPL_H_
#define ARCHITECTURE_TENSORIMPL_H_

#include <iostream>
#include <vector>
#include <initializer_list>
#include <type_traits>
#include <memory>
#include <stdexcept>
#include <cstring>
#include <sstream>
#include "DarkMagic.h"
#include "../tools/FileSystemTools.h"



/**
 * Provide shape information for Tensors.
 *
 * Shapes that are attached to Tensors should be treated as immutable.
 * A lot of things will break if the shape is changed.
 *
 */
class Shape {
public:
	size_t size;
	Shape( std::initializer_list<size_t> l )
			: size( l.size() ), shapeVector( l ) {
		mCapacity = 1;
		for ( auto v : l )
			mCapacity *= v;
	}

	Shape( std::vector<size_t> l )
			: size( l.size() ), shapeVector( l ) {
		mCapacity = 1;
		for ( auto v : l )
			mCapacity *= v;
	}

	Shape( std::vector<unsigned int> l )
			: size( l.size() ){
		for( auto one: l )
			shapeVector.push_back( one );
		mCapacity = 1;
		for ( auto v : l )
			mCapacity *= v;
	}

	// copy cotr
	Shape( const Shape& other )
		: size( other.size ), mCapacity( other.mCapacity ),	shapeVector( other.shapeVector ) {
	}

	virtual ~Shape() {
	}

	/**
	 * Returns the number of elements a that can be stored in a Tensor of this shape
	 */
	size_t capacity() {
		return mCapacity;
	}

	// operators
	bool operator==( Shape& other ) {
		if ( shapeVector.size() != other.shapeVector.size() )
			return false;
		for ( size_t i = 0; i < shapeVector.size(); ++i )
			if ( this->shapeVector [ i ] != other [ i ] )
				return false;
		return true;
	}

	bool operator!=( Shape& other ) {
		return ! ( *this == other );
	}


	const size_t& operator[]( size_t idx ) const {
		return shapeVector [ idx ];
	}

	size_t& operator[]( size_t idx ) {
		return shapeVector[ idx ];
	}

	// TODO make me pretty
	friend std::ostream& operator<<( std::ostream& output, const Shape& s ) {
		output << "[";
		for ( auto i : s.shapeVector )
			output << i << ", ";
		output << "]";
		return output;
	}

	/**
	 * Recompute the the capacity of Shape. Needs to be called if a shape is ever modified
	 */
	void computeCapacity() {
		mCapacity = 1;
		for ( auto v : shapeVector )
			mCapacity *= v;
	}

private:
	size_t mCapacity;
	std::vector<size_t> shapeVector;

};


/**
 * Abstract base for all Tensors.
 *
 * A tensor provides a view to storage object that allows us to reshape the view.
 * The shape of an tensor indicates how the data can be accessed.
 *
 * Tensors provide access via the [] operator. Since it allows for multiple indices
 * to be passed Tensors not be acces with initializer lists e.g t[ { 1,2,3 } ].
 *
 * Currently is not possible to access subtensors i.e do slicing of tensors. This
 * means in turn that during access an index for every dimension must be provided.
 *
 *
 * The storage of tensor is not initialized upon the creation of the tensor and needs
 * to be manually initialized by calling one fo the `init` functions.
 *
 */

template<class ValueType>
class Tensor {
public:

	Shape shape;

	Tensor( Shape s ) :
			Tensor( s, { }, { } ) {
		// 		rather than passing starts/ends, pass in an empty vector,
		//		otherwise, we have undefined behavior
	}

	/** @brief
	 * Constructor that uses the them same underlying data as the input tensor
	 */
	Tensor( Shape s, std::shared_ptr<Tensor<ValueType>> other )
			: Tensor( s ) {
		auto datapr = other->mdata;
		this->mdata = datapr;
		mStorageCreated = true;
		computeStrides();
	}

	Tensor( const Tensor& other ) = delete;
	void operator=( const Tensor & other ) = delete;

	virtual ~Tensor() {
	}

	// currently not used anyway
//	// operators for numbers
//	template<class T>
//	Tensor operator+( T val );
//	template<class T>
//	Tensor operator-( T val );
//	template<class T>
//	Tensor operator*( T val );

//	template<class T>
//	Tensor operator/=( T val ){
//		for ( uint i = 0; i < shape.capacity(); ++i ) {
//			*mdata[ i ] /= val;
//		}
//	}
//
//	// operators for other tensors
//	Tensor operator+( Tensor other );
//	Tensor operator-( Tensor other );
//	Tensor operator*( Tensor other );

	// access operator
	virtual ValueType& operator[]( std::initializer_list<size_t> idxs ) {
		// bounds check
		boundsCheck( idxs );
		return mdata.get()[ toIndex( idxs ) ];
	}

	/** @brief Provides access to tensor as if it were flattened.
	 * Can be negative. In this case it follows python logic and accesses the
	 * Tensor from the back.
	 *
	 */
	virtual ValueType& operator[]( long idx ) {
		// bounds check
		size_t _idx;
		if( idx < 0 )
			_idx = this->shape.capacity() + idx;
		else
			_idx = unsigned( idx );
		boundsCheck( _idx );
		return mdata.get()[ _idx ];
	}

	bool operator==( Tensor& other ) { // TODO compare arrays of different data types
		// bounds check
		if ( shape != other.shape )
			return false;
		for ( size_t i = 0; i < shape.capacity(); ++i ) {
			if ( mdata.get()[ i ] != other.mdata.get()[ i ] )
				return false;
		}
		return true;
	}

	virtual std::vector<int> argmaxVector( uint axis=0 ) = 0;

	/**
	 * Used to feed data to a tensor with already initailized storage
	 */
	template<class T>
	void feed( T& vector ) {
		if ( !mStorageCreated )
			throw std::logic_error( "Tensor storage not initialized" );
		init( vector );
	}
	
	void feed( Tensor<ValueType>& other ){
		if ( !mStorageCreated )
			throw std::logic_error( "Tensor storage not initialized" );
		if ( this->shape.capacity() != other.shape.capacity() ) {
			std::cerr << "Tensor shapes incompatible.";
			exit( 1 );
		}
		auto thisStorage = mdata.get();
		auto otherSorage = other.mdata.get();
		
		for( size_t i=0; i< this->shape.capacity(); ++i )
			thisStorage[ i ] = otherSorage[ i ];

	}

	/**
	 * Reshapes the tensor. It does not alter the storage of the tensor or any other
	 * views of this tensors.
	 *
	 * The target shape needs to be off the same capacity.
	 */
	void reshape( Shape newShape ) {
		if ( shape.capacity() != newShape.capacity() )
			throw std::logic_error( "can not reshape into shape with different capacity" );
		this->shape = newShape;
		computeStrides();
	}

	/**
	 * Special reshape function. Reshapes into a 1D tensor of same capacity
	 */
	void flatten() {
		this->reshape( Shape( { this->shape.capacity() } ) );
	}

	virtual ValueType empty() = 0;

	/**
	 * @brief A hook that can be called to perform various checks on the content
	 * of the vector. Gets called at the end of each layer. One of its uses is to perform
	 * denoising of the ciphertexts if the noise grows out of control.
	 *
	 */
	virtual void performChecks(){}; // default does nothing


	/*
	 *
	 * Memory allocation methods
	 *
	 */

	// allocates the memory and fills it with zeros
	virtual void init() = 0;

	/**
	 * @brief
	 *  allocates the memory and fills it with random values [0,1]
	 *  You can pass a seed for the random generator. Default seed is 7
	 */
	virtual void initRandom( uint seed=7 ) = 0;

	template< class T >
	void init(T& in){
		writeIndex = 0;
		init_internal( in );
	}

	// reads in data from a vector
	// TODO ? dynamically find out nested depth
	// this is not the greatest way. it does not allow for
	// any arbitraty vectors
	virtual void init( std::vector<ValueType>& in ) {
		if ( in.size() != shape [ 0 ] ) {
			std::stringstream out;
			out << "vector with shape: [";
			out << in.size() << " ]";
			out << " does not fit into  tensor with shape " << shape;
			throw std::logic_error( out.str() );
		}
		writeIndex = 0;
		init_internal( in );
	}

	virtual void init( std::vector<std::vector<ValueType>>& in ) {
		if ( in.size() != shape [ 0 ] || in [ 0 ].size() != shape [ 1 ] ) {
			std::stringstream out;
			out << "vector with shape: [";
			out << in.size() << ", " << in [ 0 ].size() << ", " << " ]";
			out << " does not fit into  tensor with shape " << shape;
			throw std::logic_error( out.str() );
		}
		writeIndex = 0;
		for ( auto& one : in )
			init_internal( one );
	}

	virtual void init( std::vector<std::vector<std::vector<ValueType>>>& in ) {
		if ( in.size() != shape [ 0 ] || in [ 0 ].size() != shape [ 1 ] || in [ 0 ] [ 0 ].size() != shape [ 2 ] ) {
			std::stringstream out;
			out << "vector with shape: [";
			out << in.size() << ", " << in [ 0 ].size() << ", " << in [ 0 ] [ 0 ].size() << " ]";
			out << " does not fit into  tensor with shape " << shape;
			throw std::logic_error( out.str() );
		}
		writeIndex = 0;
		for ( auto& one : in )
			init_internal( one );
	}

	virtual void init( std::vector<std::vector<std::vector<std::vector<ValueType>>>>& in ) {
		if ( in.size() != shape [ 0 ] || in [ 0 ].size() != shape [ 1 ] || in [ 0 ] [ 0 ].size() != shape [ 2 ]
				|| in [ 0 ] [ 0 ] [ 0 ].size() != shape [ 3 ] ) {
			std::stringstream out;
			out << "vector with shape: [";
			out << in.size() << ", " << in [ 0 ].size() << ", " << in [ 0 ] [ 0 ].size() << ", "
					<< in [ 0 ] [ 0 ] [ 0 ].size() << " ]";
			out << " does not fit into  tensor with shape " << shape;
			throw std::logic_error( out.str() );
		}
		writeIndex = 0;
		for ( auto& one : in )
			init_internal( one );
	}

//	// creates a clone of the tensor
//	Tensor<ValueType> clone() {
//		Tensor<ValueType> ret( shape );
//		this->copyInto( ret );
//		return ret;
//	}

	// copys this tensor into a target tensor
	void copyInto( Tensor<ValueType>& target ) {
		static_assert(shape==target.shape, "");
		target.createStorage(); // make sure the target storage is ready
		std::memcpy( mdata, target.mdata, shape.mCapacity * sizeof(ValueType) );
	}

	void fillString( std::string &nestedString, std::string& arrayString,
			uint& copy_idx,
			uint& val_per_line ) {

		int counter = 1; // hacky way to keep track of the commas. must start @ 1.
		for ( size_t i = 0; i < shape.capacity(); ++i ) {
			arrayString += std::to_string( mdata.get() [ i ] );
			if ( counter < val_per_line ) {
				arrayString += ",";
				counter++;
			}
			//insert arrayString with
			if ( i != 0 && ( i + 1 ) % val_per_line == 0 ) {
				copy_idx++;
				//TODO: delete last "," before appending to nestedString
				nestedString.append( "[" + arrayString + "]" + "\n" );
				arrayString.clear();
				counter = 1;
			}
		}
	}



	// The recurse through T and put values into string creating nested representation
	void createStringReper( std::string& nestedString, uint& braces ) {
		std::cout << "Number of values to be used for nested string representation: "
				<< shape.capacity() << std::endl;

		uint copy_idx = braces; // number of open and closing braces
		std::string arrayString = ""; //tmp string to hold this dims values

		// put opening braces
		while ( copy_idx > 0 ) {
			nestedString += '[';
			--copy_idx;
		}

		fillString( nestedString, arrayString, copy_idx, braces );

		// closing braces
		while ( copy_idx > 0 ) {
			nestedString += ']';
			--copy_idx;
		}
//		assert( copy_idx == shape.capacity() ); //means every value in tensor was vistited and copied

//		nestedString += ']'; //put last closing bracket

	}


	/**
	* Writes the Tensor conent flattened to a binary file.
	*/
	virtual void writeToFile( std::string file ) = 0;
	
	void writeToFile( std::string path, std::string file ){
		writeToFile( joinPath( path, file ) );
	}
	
	


	friend std::ostream &operator<<( std::ostream &output, Tensor &T ) {
		ValueType* temp = T.mdata.get();
		output << "[";
		for ( uint i = 0; i < T.shape.capacity(); ++i ) {
			for( size_t s =0; s < T.strides.size() - 1; ++s ){
				if( i != 0 && i % T.strides[s] == 0 ){
					output << std::endl;
					break;
				}
			}
			output << temp[ i ];
			if ( i != T.shape.capacity() - 1 )
				output << ", ";
		}
		output << std::endl << "shape: " << T.shape;
		return output;
	}
	
	
	void clear(){
		mdata = NULL;
		mStorageCreated = false;
	}


//	//FIXME misses all checks
//	void transpose( const std::vector<size_t>& perm ){
//		// needs to be moved around. cant work with strides
//	}

	std::shared_ptr<ValueType> mdata; // FIXME just for debugging, should be protected
protected:
	std::vector<uint> strides;
	bool mStorageCreated = false;
	// reshape constructor


	/**
	 * precompute the strides for index into the storage array
	 * every index is multplied with the product of the following dimensions
	 * considering a tensor with shape( x,y,z ) the index i would be calculated as
	 * ( i1 * ( y * z )) + ( i2 * z ) + i3
	 */
	virtual void computeStrides() {
		strides.clear();
		for ( size_t dim = 0; dim < shape.size; ++dim ) {
			uint temp = 1;
			for ( size_t i = dim + 1; i < shape.size; ++i )
				temp *= shape [ i ];
			strides.push_back( temp );
		}
	}


	void boundsCheck( std::initializer_list<size_t>& idxs ) {
		if ( starts.empty() ) {
			if ( idxs.size() != shape.size ) {
				boundsErrror( idxs );
			}
			// TODO this code nees to be as fast possible
			//could remove function call to expand() and instead do func inline, would be very minimal savings
			if ( expand( idxs ) >= shape.capacity() ) {
				boundsErrror( idxs );
			}
			size_t i = 0;
			for ( auto one : idxs ) {
				if ( one >= shape [ i++ ] ) {
					boundsErrror( idxs );
				}
			}
		} else {
			// TODO slicing
		}
	}

	size_t toIndex( const std::initializer_list<size_t> & idxs ) {
		// every index is multiplied with the product of the following dimensions
		// considering a tensor with shape( x,y,z ) the index i would be calculated as
		// ( i1 * ( y * z )) + ( i2 * z ) + i3
		size_t idx = 0;
		size_t i = 0;
		for ( auto x : idxs )
			idx += x * strides [ i++ ];
		return idx;
	}

	// we need this to use shared pointers with the storage array
	struct array_deleter {
		void operator ()( ValueType const * p ) {
			delete [ ] p;
		}
	};

	// create the storage array
	// IT DOES NOT INITILIZE THE VALUES
	// if memory has been allocated it is a no op
	virtual void createStorage() =0;


private:
	// for slices
	std::vector<uint> starts;
	std::vector<uint> ends;


	// only used during writing values into the tensor
	uint writeIndex = 0;

	Tensor( Shape s, std::vector<uint> starts, std::vector<uint> ends )
			: shape( s ), starts( starts ), ends( ends ) {
		computeStrides();
	}

	Tensor( Shape s, const Tensor& t, std::vector<uint> starts, std::vector<uint> ends )
			: Tensor( s, starts, ends ) {
		mdata = t.mdata;
	}


	void boundsErrror( std::initializer_list<size_t>& idxs ) {
		std::vector<size_t> vec( idxs );
		std::stringstream out;
		for ( auto i : vec )
			out << i << ", ";
		out << " for tensor with shape " << shape;
		throw std::out_of_range( out.str() );
	}

	void boundsCheck( size_t idx ) {
		if ( starts.empty() ) {
			if ( idx >= shape.capacity() ){
				std::stringstream out;
				out << idx << " out of range for tensor with capacity " << shape.capacity();
				throw std::out_of_range( out.str() );
			}
		} else {
			// TODO slicing
		}
	}

	template<class T>
	static uint expand( const T& values ) {
		uint result = 1;
		for ( auto i : values )
			result *= i;
		return result;
	}

	template<class T>
	void init_internal( std::vector<ValueType>& in ) {
		if( is_vector<T>::value  ){ // dealing with a vector
			for( auto inner: in ){
				init_internal( in );
			}
		}
	}

	void init_internal( std::vector<ValueType>& in ) {
		createStorage();
		for ( auto one : in )
			mdata.get() [ writeIndex++ ] = one;
	}

	void init_internal( std::vector<std::vector<ValueType>>& in ) {
		createStorage();
		for ( auto& one : in )
			init_internal( one );
	}

	void init_internal( std::vector<std::vector<std::vector<ValueType>>>& in ) {
		createStorage();
		for ( auto& one : in )
			init_internal( one );
	}

	void init_internal( std::vector<std::vector<std::vector<std::vector<ValueType>>>>& in ) {
		createStorage();
		for ( auto& one : in )
			init_internal( one );
	}


};


/// For the lazy
template<class T>
using TensorP = std::shared_ptr<Tensor<T>>;


template<class T>
std::vector<T> range( T end, T increment = 1 ) {
	std::vector<T> vec;
	for ( T i = 0; i < end; i += increment ) {
		vec.push_back( i );
	}
	return vec;
}

template<class T>
Shape getShapeFromVector( T& in, bool strictDimensionChecking=true ){
	std::vector<uint> vec;
	getShapeFromVectorImpl( in, vec, strictDimensionChecking );
	return Shape(vec);
}

template<class T>
void getShapeFromVectorImpl( T& in, std::vector<uint>& vec, bool strictDimensionChecking ){
	if( is_vector<T>::value ){
		vec.push_back( in.size() );
	}
	//TODO strict dimension checking
	if( is_vector<typename decltype( in[0] )::value_type>::value )
		getShapeFromVectorImpl( in[ 0 ], vec, is_vector<typename decltype( in[0] )::value_type>::value, strictDimensionChecking );
}





#endif /* ARCHITECTURE_TENSORIMPL_H_ */
