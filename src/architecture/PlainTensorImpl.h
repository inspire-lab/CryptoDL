/*
 * PlainTensorImpl.h
 *
 *  Created on: Feb 18, 2019
 *      Author: robert
 */

#ifndef PLAINTENSORIMPL_H_
#define PLAINTENSORIMPL_H_

#include <type_traits>
#include <random>
#include "Tensor.h"
#include "TensorFactory.h"

template<class ValueType>
class PlainTensor: public Tensor<ValueType> {
public:

	PlainTensor( Shape s )
			: Tensor<ValueType>::Tensor( s ) {
	}

	PlainTensor( Shape s, TensorP<ValueType> other )
			: Tensor<ValueType>::Tensor( s, other ) {
	}

	// operators for numbers
	template<class T>
	PlainTensor<ValueType> operator+( T val ) {
		for ( size_t i = 0; i < this->shape.capacity(); ++i )
			this->mdata[ i ] += val;
		return this;
	}
	template<class T>
	PlainTensor<ValueType> operator-( T val ) {
		for ( size_t i = 0; i < this->shape.capacity(); ++i )
			this->mdata[ i ] -= val;
		return this;
	}
	template<class T>
	PlainTensor<ValueType> operator*( T val ) {
		for ( size_t i = 0; i < this->shape.capacity(); ++i )
			this->mdata[ i ] *= val;
		return this;
	}

	// operators for other tensors
	template<class T>
	PlainTensor<ValueType> operator+( Tensor<T> other ) {
		static_assert( this->shape == other.shape(), " " );
		for ( size_t i = 0; i < this->shape.capacity(); ++i )
			this->mdata[ i ] += other.data[ i ];
		return this;

	}
	template<class T>
	PlainTensor<ValueType> operator-( Tensor<T> other ) {
		static_assert( this->shape == other.shape(), " " );
		for ( size_t i = 0; i < this->shape.capacity(); ++i )
			this->mdata[ i ] -= other.data[ i ];
		return this;
	}
	template<class T>
	PlainTensor<ValueType> operator*( Tensor<T> other ) {
		static_assert( this->shape == other.shape(), " " );
		for ( size_t i = 0; i < this->shape.capacity(); ++i )
			this->mdata[ i ] *= other.data[ i ];
		return this;
	}

	virtual ValueType empty() override {
		return 0;
	}

	virtual std::vector<int> argmaxVector( uint axis = 0 )  override {
		std::vector<int> ret;
		//TODO argmax currently only works on the last dimension and for 2D tensors
		for ( uint i = 0; i < this->shape[ 0 ]; ++i ) {
			ValueType max = 0;
			bool valueSet = false;
			int clazz = 0;
			for ( uint j = 0; j < this->shape [ 1 ]; ++j ) {
				if ( !valueSet || max < this->operator[]( { i, j } ) ) {
					max = this->operator[]( { i, j } );
					clazz = j;
					valueSet = true;
				}
			}
			ret.push_back( clazz );
		}
		return ret;
	}

	virtual void init() override {
		this->createStorage();
		for ( size_t i = 0; i < this->shape.capacity(); ++i )
			this->mdata.get()[ i ] = static_cast<ValueType>( 0 );
	}


	virtual void initRandom( uint seed=7 ) override {
		initRandomInternal( seed, std::is_floating_point<ValueType>() );
	}

	// this is used for real numbers
    void initRandomInternal( uint seed, std::true_type ){
    	std::mt19937 mt( seed ); // seed the generator
		std::uniform_real_distribution<ValueType> dist( 0, 1 );
		createStorage();
		for( uint i=0; i < this->shape.capacity(); ++i  )
			this->mdata.get() [ i ] = dist( mt );
    }


	void initRandomInternal( uint seed, std::false_type ){
		std::mt19937 mt( seed ); // seed the generator
		std::uniform_int_distribution<ValueType> dist( 0, 1 );
		createStorage();
		for( uint i=0; i < this->shape.capacity(); ++i  )
			this->mdata.get() [ i ] = dist( mt );
	}

	virtual void createStorage() override {
		if ( this->mStorageCreated )
			return;
		this->mdata = std::shared_ptr<ValueType>( new ValueType [ this->shape.capacity() ],
				std::default_delete<ValueType [ ]>() );
		this->mStorageCreated = true;
	}
	
	virtual void writeToFile( std::string file ){
		std::fstream myfile;
		myfile = std::fstream( file, std::ios::out | std::ios::binary );
		if ( !myfile ){
			std::cerr << "something messed up when opening " << file << std::endl;
			exit( 1 );
		}
		myfile.write( (char*) this->mdata.get(), sizeof( ValueType ) * this->shape.capacity() );		
		myfile.close();
	}


	virtual ~PlainTensor() {
	}
};


template<class T>
class PlainTensorFactory: public TensorFactory<T> {
public:
	PlainTensorFactory() {
	}

	virtual TensorP<T> create( Shape s ) override {
		return std::make_shared<PlainTensor<T>>( s );
	}

	virtual TensorP<T> createView( Shape s, TensorP<T> other ) override {
		return std::make_shared<PlainTensor<T>>( s, other );
	}

	TensorP<T> ones( Shape s ) {
		TensorP<T> tensor = std::make_shared<PlainTensor<T>>( s );
		std::vector<T> data( s.capacity(), 1 );
		tensor->flatten();
		tensor->init( data );
		tensor->reshape( s );
		return tensor;
	}

	TensorP<T> zeros( Shape s ) {
		TensorP<T> tensor = std::make_shared<PlainTensor<T>>( s );
		tensor->init();
		return tensor;
	}

	TensorP<T> range( Shape s ) {
		std::vector<T> data( s.capacity(), 1 );
		for ( T i = 0; i < data.size(); ++i )
			data[ i ] = i;
		TensorP<T> tensor = std::make_shared<PlainTensor<T>>( s );
		tensor->flatten();
		tensor->init( data );
		tensor->reshape( s );
		return tensor;

	}

	virtual ~PlainTensorFactory() {
	}

	//numpy-like funcs (FIXME: They assume shape of size 4)
	virtual TensorP<T> onesAndInit( Shape s ) {
		TensorP<T> ones = std::make_shared<PlainTensor<T>>( s );

		std::vector<std::vector<std::vector<std::vector<T>>>> onesV =
				std::vector<std::vector<std::vector<std::vector<T>>>>(
						s [ 0 ],
						std::vector<std::vector<std::vector<T>>>( s [ 1 ],
								std::vector<std::vector<T>>( s [ 2 ],
										std::vector<T>( s [ 3 ], 1 ) ) ) );

		ones->init( onesV );

		return ones;
	}

	virtual TensorP<T> arangeAndInit( Shape s, int start = 1, int step = 1 ) {
		//create a 1d tensor that can holds a*b*c*d values
		unsigned int size = s [ 0 ] * s [ 1 ] * s [ 2 ] * s [ 3 ];
		TensorP<T> range = std::make_shared<PlainTensor<T>>( Shape { size } );

		//create a vector of constantly increasing values based off step
		std::vector<T> values;
		int curValue = start;
		for ( unsigned int i = 0; i < size; i++ ) {
			values.push_back( (T) curValue ); //todo: ensure this type of cast is ok
			curValue += step;
		}

		range->init( values );

		//reshape tensor
		range->reshape( s );

		return range;
	}
};


#endif /* PLAINTENSORIMPL_H_ */
