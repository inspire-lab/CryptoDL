/*
 * CipherTextWrapper.h
 *
 *  Created on: Mar 2, 2019
 *      Author: robert
 */

#ifndef ARCHITECTURE_HEBACKEND_CIPHERTEXTWRAPPER_H_
#define ARCHITECTURE_HEBACKEND_CIPHERTEXTWRAPPER_H_


#include "../Tensor.h"


template<class T>
class HETensorFactory;

template<class CiphterTextWrapper>
class CipherTextWrapperFactory {
public:


	virtual void setAsDefaultFactory() = 0;

	virtual CiphterTextWrapper empty() = 0;

	virtual CiphterTextWrapper createCipherText( long x )= 0;

	virtual CiphterTextWrapper createCipherText( const std::vector<long>& in ) =0;

	virtual CiphterTextWrapper createCipherText( const std::vector<double>& in ) =0;

	virtual CiphterTextWrapper createCipherText( const std::vector<float>& in ) =0;

	virtual std::vector<long> decryptLong( const CiphterTextWrapper& ctx ) = 0;

	virtual std::vector<double> decryptDouble( const CiphterTextWrapper& ctx ) = 0;

	virtual uint batchsize()=0;

	/**
	 * @brief Take a 1D vector transform into the given shape and encrypt it. The 1st dimension of the same needs to line up with the batchSize
	 * supported by the encryption scheme.
	 */
	virtual TensorP<CiphterTextWrapper> createCipherTensor( const std::vector<double>& in, const Shape& shape, HETensorFactory<CiphterTextWrapper>* hetf ) = 0;

	virtual TensorP<CiphterTextWrapper> createCipherTensor( const std::vector<float>& in, const Shape& shape, HETensorFactory<CiphterTextWrapper>* hetf ) = 0;

	/**
	 * Encrypts the data and sticks it into the ciphertensor.
	 * If batchSize == -1 the batch size is infered from the crypto parameters otherwise it
	 * must not be larger than the batch size infered from the crypto parameters. 
	 */
	virtual void feedCipherTensor( const std::vector<double>& in, TensorP<CiphterTextWrapper> tensor, int batchSize=-1 ) = 0;

	/**
	 * Encrypts the data and sticks it into the ciphertensor.
	 * If batchSize == -1 the batch size is infered from the crypto parameters otherwise it
	 * must not be larger than the batch size infered from the crypto parameters. 
	 */
	virtual void feedCipherTensor( const std::vector<float>& in, TensorP<CiphterTextWrapper> tensor, int batchSize=-1 ) = 0;

	virtual void feedCipherTensor( const TensorP<double> in, TensorP<CiphterTextWrapper> tensor ) = 0;

	virtual void feedCipherTensor( const TensorP<double> in, Tensor<CiphterTextWrapper>& tensor ) = 0;

	virtual ~CipherTextWrapperFactory() {
	}

	void enableRefreshOnHighNoise(){
		this->mRefreshOnHighNoise = true;
	}

	void disableRefreshOnHighNoise(){
		this->mRefreshOnHighNoise = true;
	}

	bool refreshOnHighNoiseEnabled(){
		return mRefreshOnHighNoise;
	}



private:
	bool mRefreshOnHighNoise = false;

};





//template<class T>
//CipherTextWrapper* operator*( CipherTextWrapper* ctw, T& other ) {
//	std::cout << "WE NEED TO LOOK AT MEMORY MANAGEMENT" << std::endl;
//	CipherTextWrapper* ret = ctw->empty();
//	*ret *= other;
//	return ret;
//}

//template<class T>
//CipherTextWrapper* & operator( CipherTextWrapper* ctw, T& other ) {
//	std::cout << "WE NEED TO LOOK AT MEMORY MANAGEMENT" << std::endl;
//	CipherTextWrapper* ret = ctw->empty();
//	*ret *= other;
//	return ret;
//}





#endif /* ARCHITECTURE_HEBACKEND_CIPHERTEXTWRAPPER_H_ */
