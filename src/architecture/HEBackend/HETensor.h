/*
 * HETensor.h
 *
 *  Created on: Mar 3, 2019
 *      Author: robert
 */

#ifndef ARCHITECTURE_HEBACKEND_HETENSOR_H_
#define ARCHITECTURE_HEBACKEND_HETENSOR_H_

#include <type_traits>
#include <iostream>
#include <fstream>
#include "CipherTextWrapper.h"
#include "../Tensor.h"
#include "../PlainTensor.h"

template<class T>
class HETensorFactory;

template<class T>
class HETensor: public Tensor<T> {
public:

	HETensor( Shape s, Shape plainTextShape, CipherTextWrapperFactory<T>* factory )
			: Tensor<T>::Tensor( s ), mPlainTextShape( plainTextShape ), mFactory( factory ) {
		computeStrides();
	}

	HETensor( Shape s, Shape plainTextShape, CipherTextWrapperFactory<T>* factory, TensorP<T> other )
			: Tensor<T>::Tensor( s, other ), mPlainTextShape( plainTextShape ), mFactory( factory ) {
		computeStrides();
	}


	virtual ~HETensor() {
	}

	static std::shared_ptr<HETensor<T>> createTensor( Shape s, Shape plainTextShape,
			CipherTextWrapperFactory<T>* factory ) {
		return std::make_shared<HETensor<T>>( s, plainTextShape, factory );
	}


	/**
	 * @brief Runs some checks on the content of the tensor.
	 * If the factory has `refreshOnHighNoiseEnabled` this triggers
	 * the data to be sent to the client for rencryption.
	 */
	virtual void performChecks() override {
		if( mFactory->refreshOnHighNoiseEnabled() ){
			for( size_t i = 0; i < this->shape.capacity(); ++ i ){
				if( this->mdata.get()[ i ].noiseNearOverflow() || true ){
					// std::cout << "WARNING checks for noiseoverflow not performed correctly" << std::endl;
					this->writeToFile( "old_" + std::to_string( mFactory->batchsize() ) + ".bin" )  ;
					refreshCipherTexts();
					this->writeToFile( "new_" + std::to_string( mFactory->batchsize() ) + ".bin" )  ;
					break;
				}
			}
		}
		
	};


	TensorP<double> decryptDouble() {
		Shape oldShape = this->shape; // @suppress("Invalid arguments")
		// flatten the vector
		this->reshape( { this->shape.capacity() } );

		// decrypt the pixel batch
		// the shape of this is [ px, batch ]
		// we want [ batch, px  ] though
		std::vector<std::vector<double>> plainVector( this->mPlainTextShape[ 0 ],
				std::vector<double>( this->shape.capacity(), 0 ) );
		for ( uint px = 0; px < this->shape.capacity(); ++px ) {
			std::vector<double> inner = this->mFactory->decryptDouble( this->operator []( { px } ) ); // @suppress("Invalid arguments")
			for ( uint batch = 0; batch < this->mPlainTextShape[ 0 ]; ++batch ) {
				plainVector[ batch ][ px ] = inner[ batch ];
			}
		}

		// restore the old shape
		this->reshape( oldShape );

		//create plain tensor with same shape as the plain vector
		TensorP<double> ret = PlainTensorFactory<double>().create( Shape {	plainVector.size(), plainVector [ 0 ].size() } ); // @suppress("Symbol is not resolved")
		ret->init( plainVector );

		// bring the plaintensor into the correct shape
		ret->reshape( mPlainTextShape );
		return ret;
	}

	TensorP<long> decryptLong() {
		Shape oldShape = this->shape; // @suppress("Invalid arguments")
		// flatten the vector
		this->reshape( { this->shape.capacity() } );

		// decrypt the pixel batch
		// the shape of this is [ px, batch ]
		// we want [ batch, px  ] though
		uint lastDimSize = oldShape.size == 4 ? oldShape[ 2 ] * oldShape[ 3 ] : oldShape[ oldShape.size - 1 ];
		std::vector<std::vector<long>> plainVector( mPlainTextShape [ 0 ], std::vector<long>( lastDimSize, 0 ) );
		for ( size_t px = 0; px < this->shape.capacity(); ++px ) {
			auto inner = this->mFactory->decryptLong( this->operator []( { px } ) );
			for ( uint batch = 0; batch < this->mPlainTextShape [ 0 ]; ++batch ) {
				plainVector [ batch ] [ px ] = inner [ batch ];
			}
		}

		// restore the old shape
		this->reshape( oldShape );

		//create plain tensor with same shape as the plain vector
		TensorP<long> ret = PlainTensorFactory<long>().create( Shape{ (uint) plainVector.size(), (uint) plainVector [ 0 ].size() } ); // @suppress("Symbol is not resolved")
		ret->init( plainVector );

		// bring the plaintensor into the correct shape
		ret->reshape( mPlainTextShape );
		return ret;
	}


	virtual T empty() override {
		return mFactory->empty();
	}

	virtual std::vector<int> argmaxVector( uint axis = 0 ) override {
		throw std::logic_error( "Cant do argmax on HETensor" );
	}


	virtual void init() override {
		createStorage();
//		std::cout << "creating storage" << std::endl;
		for ( uint i = 0; i < this->shape.capacity(); ++i )
			this->mdata.get()[ i ] = this->mFactory->empty();
	}

	virtual void initRandom( uint seed=7  ) override {
		throw std::logic_error( "init random not implemented for HETensors yet" );
	}

	/**
	 * @brief Refreshes the ciphertexts in the tensor.
	 * Currently they are decrypted and rencrypted. A more realistic
	 * implementation would send them to the holder of the private key.
	 *
	 */
	void refreshCipherTexts(){
		std::cout << "refreshing ciphertexts..";
		auto decrypted = decryptDouble();
		// std::cout << *decrypted << std::endl;
		mFactory->feedCipherTensor( decrypted, *this );
		std::cout << "done" << std::endl;
	}


	virtual void writeToFile( std::string file ){
		std::fstream myfile;
		myfile = std::fstream( file, std::ios::out | std::ios::binary );
		if ( !myfile ){
			std::cerr << "something messed up when opening " << file << std::endl;
			exit( 1 );
		}
		for( size_t i = 0; i < this->shape.capacity(); ++i )
			this->mdata.get()[ i ].writeToFile( myfile );
		myfile.close();
	}

	CipherTextWrapperFactory<T>* mFactory;
	
protected:
	Shape mPlainTextShape;


	/**
	 * Strides and shapes work somewhat differently with HE Tensors. Since we
	 * are working with SIMD our shape is [ channel, y, x, batch ]. This also changes
	 * the capacity of.
	 *
	 * We want to keep the same shape notation as regular tensors.
	 */
	virtual void computeStrides() override {
		Tensor<T>::computeStrides();
		// we operate like we only have one batch
		this->strides[ 0 ] = 1;
	}

	virtual void createStorage() override {
		if ( this->mStorageCreated )
			return;
		if(! T::defaultFactory)
			mFactory->setAsDefaultFactory() ;
		this->mdata = std::shared_ptr<T>( new T [ this->shape.capacity() ], std::default_delete<T [ ]>() );
		for ( uint i = 0; i < this->shape.capacity(); ++i )
			this->mdata.get() [ i ] = this->mFactory->empty();
		this->mStorageCreated = true;

	}

};

template<class T>
using HETensorP = std::shared_ptr<HETensor<T>>;

template<class T>
class HETensorFactory: public TensorFactory<T> {
public:
	HETensorFactory( CipherTextWrapperFactory<T>* factory )
			: mFactory( factory ) {
	}

	virtual TensorP<T> create( Shape s ) override {
		Shape tensorShape = s;
		tensorShape [ 0 ] = 1; // rethink this
		/// Store the plain text shape for convience
		Shape plainTextShape = s;
		plainTextShape [ 0 ] = mFactory->batchsize();
		plainTextShape.computeCapacity();
		return std::make_shared<HETensor<T>>( tensorShape, plainTextShape, mFactory );
	}

	virtual TensorP<T> createView( Shape s, TensorP<T> other ) override {
		Shape tensorShape = s;
		tensorShape [ 0 ] = 1; // rethink this
		/// Store the plain text shape for convience
		Shape plainTextShape = s;
		plainTextShape [ 0 ] = mFactory->batchsize();
		plainTextShape.computeCapacity();
		return std::make_shared<HETensor<T>>( tensorShape, plainTextShape, mFactory, other );

	}

	virtual ~HETensorFactory() {
	}

	CipherTextWrapperFactory<T>* ciphertextFactory() const {
		return mFactory;
	}

	//TODO: fix up for encrypted values and assuming shape size 4 (do not want that)
	virtual TensorP<T> onesAndInit( Shape s ) {
		TensorP<T> ones;		// = std::make_shared<HETensor<T>>( s );

		//may need to encrypt 1
		/*std::vector<std::vector<std::vector<std::vector<T>>>> onesV =
		 std::vector<std::vector<std::vector<std::vector<T>>>>(
		 s [ 0 ],
		 std::vector<std::vector<std::vector<T>>>( s [ 1 ],
		 std::vector<std::vector<T>>( s [ 2 ],
		 std::vector<T>( s [ 3 ], 1 ) ) ) );

		 ones->init( onesV );*/

		return ones;
	}

	virtual TensorP<T> arangeAndInit( Shape s, int start = 1, int step = 1 ) {
		//create a 1d tensor that can holds a*b*c*d values
		//unsigned int size = s [ 0 ] * s [ 1 ] * s [ 2 ] * s [ 3 ];
		TensorP<T> range;// = std::make_shared<HETensor<T>>( Shape( { size } ) );

		/*
		//create a vector of constantly increasing values based off step
		std::vector<T> values;
		int curValue = start;
		 T curValue = encrypt(start);
		 T eStep = encrypt(step);
		for ( unsigned int i = 0; i < size; i++ ) {
		 values.push_back( curValue );
		 curValue += eStep;
		}

		 range->init( values );*/

		//reshape tensor
		//range->reshape( s );

		return range;
	}

private:
	CipherTextWrapperFactory<T>* mFactory;


};


#endif /* ARCHITECTURE_HEBACKEND_HETENSOR_H_ */
