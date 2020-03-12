/*
 * HELIbCipherText.cpp
 *
 *  Created on: Mar 11, 2019
 *      Author: robert
 */



#include "HELIbCipherText.h"
#include <helib/NumbTh.h>
#include "../HETensor.h"
#include "../../PlainTensor.h"



//need to declare static memeber outside of the classs
HELibCipherTextFactory* HELibCipherText::defaultFactory = nullptr;
long HELibCipherText::noiseThreshold = NTL_OVFBND / 2.5;

HELibCipherText::HELibCipherText() :
		mFactory( defaultFactory ), mCtxt( mFactory->createRawEmpty() ) {
}

HELibCipherText HELibCipherText::empty() {
	return this->mFactory->empty();
}

HELibCipherText& HELibCipherText::operator+=( long x ) {
	if ( mFactory->useBFV )
		mCtxt->addConstant( NTL::to_ZZ( x ) );
	else
		mCtxt->addConstantCKKS( x );
	return *this;
}

HELibCipherText& HELibCipherText::operator*=( long x ) {
	if ( x == 0 ) {
		mCtxt = mFactory->createRawEmpty();
		return *this;
	}
	if ( mFactory->useBFV )
		mCtxt->multByConstant( NTL::to_ZZ( x ) );
	else
		mCtxt->multByConstantCKKS( x );
	return *this;
}

HELibCipherText& HELibCipherText::operator+=( float x ) {
	if ( mFactory->useBFV )
		mCtxt->addConstant( NTL::to_ZZ( x ) );
	else {
		mCtxt->addConstantCKKS( helib::rationalApprox( x, 1L << mCtxt->getContext().alMod.getR() ) );
	}
	return *this;
}

HELibCipherText& HELibCipherText::operator*=( float x ) {
	if ( x == 0 ) {
		mCtxt = mFactory->createRawEmpty();
		return *this;
	}
	if ( mFactory->useBFV )
		throw std::logic_error( "cant do float with bfv" );
	else
		mCtxt->multByConstantCKKS( helib::rationalApprox( x, 1L << mCtxt->getContext().alMod.getR() ) );
	return *this;
}

HELibCipherText& HELibCipherText::operator+=( double x ) {
	if ( mFactory->useBFV )
		mCtxt->addConstant( NTL::to_ZZ( x ) );
	else {
		mCtxt->addConstantCKKS( helib::rationalApprox( x, 1L << mCtxt->getContext().alMod.getR() ) );
	}
	return *this;
}

HELibCipherText& HELibCipherText::operator*=( double x ) {
	if ( x == 0 ) {
		mCtxt = mFactory->createRawEmpty();
		return *this;
	}
	if ( mFactory->useBFV )
		throw std::logic_error( "cant do float with bfv" );
	else
		mCtxt->multByConstantCKKS( helib::rationalApprox( x, 1L << mCtxt->getContext().alMod.getR() ) );
	return *this;
}




void HELibCipherTextFactory::setAsDefaultFactory(){
	HELibCipherText::defaultFactory = this;
}



HELibCipherText HELibCipherTextFactory::createCipherText( const std::vector<long> & in ) {
	std::shared_ptr<helib::Ctxt> ctxt = std::make_shared<helib::Ctxt>( *publicKey );
	if ( useBFV ) {
		ea->encrypt<std::vector<long>>( *ctxt, *publicKey, in );
	} else {
		helib::EncryptedArrayCx ea = context.get()->ea->getCx();
		ea.encrypt( *ctxt, *publicKey, in );
	}
	return HELibCipherText( ctxt, this );
}


HELibCipherText HELibCipherTextFactory::createCipherText( const std::vector<double> & in ) {
	std::shared_ptr<helib::Ctxt> ctxt = std::make_shared<helib::Ctxt>( *publicKey );
	if ( useBFV ) {
		throw std::logic_error( "cant use doubles with BFV" );
	} else {
		helib::EncryptedArrayCx ea = context.get()->ea->getCx();
		ea.encrypt( *ctxt, *publicKey, in );
	}
	return HELibCipherText( ctxt, this );
}


HELibCipherText HELibCipherTextFactory::createCipherText( const std::vector<float> & in ) {
	std::shared_ptr<helib::Ctxt> ctxt = std::make_shared<helib::Ctxt>( *publicKey );
	if ( useBFV )
		throw std::logic_error( "cant use doubles with BFV" );
	std::vector<double> doubleVector( in.begin(), in.end() );
	createCipherText( doubleVector );
	return HELibCipherText( ctxt, this );
}


std::vector<long> HELibCipherTextFactory::decryptLong( const HELibCipherText& ctx ) {
	std::vector<long> plain;
	ea->decrypt<std::vector<long>>( ctx.ctxt(), *secretKey, plain );
	return plain;
}


std::vector<double> HELibCipherTextFactory::decryptDouble( const HELibCipherText& ctx ) {
	if ( useBFV )
		throw std::logic_error( "cant decrypt doubles with BFV" );
	std::vector<double> plain( batchsize(), 0.0 );
	helib::EncryptedArrayCx ea = context.get()->ea->getCx();
	ea.decrypt( ctx.ctxt(), *secretKey, plain );
	return plain;
}

HELibCipherText& HELibCipherText::operator+=( HELibCipherText& other ) {
	*mCtxt += ( (HELibCipherText&) other ).ctxt();
	if ( mCtxt->getRatFactor().x == 0 ) {
		std::cout << "rat factor is zero" << std::endl;
//		std::cout << mFactory->decryptDouble( *this ) << std::endl;
		mCtxt = mFactory->createRawEmpty();

	}
	return *this;

}


TensorP<HELibCipherText> HELibCipherTextFactory::createCipherTensor( const std::vector<double>& in, const Shape& shape, HETensorFactory<HELibCipherText>* hetf ){
	if( shape[ 0 ] != batchsize() )
		throw std::logic_error( "Shape does not match supported batchsize" );

	const uint bs = batchsize();
	// create a vector of ciphertexts
	const size_t num = in.size() / bs; // number of elements in a batch
	std::vector<double> temp( num, 0 );

	std::vector<HELibCipherText> cipherTexts;
	for( size_t i=0; i<num; ++i )
		for( size_t batch = 0; batch < shape[ 0 ]; ++batch ){
			temp[ i ]= in[ batch + num * i ] ;
		cipherTexts.push_back( createCipherText( temp ) );
	}

	Shape newShape = shape;
	newShape[ 0 ] = 1;
	auto ret = hetf->create( { num } );

	ret->init( cipherTexts );
	ret->reshape( newShape );
	return ret;
}

// just convert to double and call the double function.

TensorP<HELibCipherText> HELibCipherTextFactory::createCipherTensor( const std::vector<float>& in, const Shape& shape, HETensorFactory<HELibCipherText>* hetf ){
	std::vector<double> doubleVector( in.begin(), in.end() );
	return createCipherTensor( doubleVector, shape, hetf );
}



void HELibCipherTextFactory::feedCipherTensor( const std::vector<double>& in, TensorP<HELibCipherText> tensor ){
	const uint bs = batchsize();
	// create a vector of ciphertexts
	std::cout << "batchsize " <<  bs << std::endl;
	std::cout << "in size " <<  in.size() << std::endl;

	const size_t num = in.size() / bs; // number of elements in a batch

	// do some sanity checking
	size_t numCheck = 1;
	for( size_t i = 1; i < tensor->shape.size; ++i )
		numCheck *= tensor->shape[ i ];
	std::cout << num << std::endl;
	std::cout << numCheck << std::endl;
	if( num != numCheck )
		throw std::logic_error( "Shape does not match supported batchsize" );


	std::vector<double> temp( bs, 0 );
	std::vector<HELibCipherText> cipherTexts;
	for( size_t i=0; i<num; ++i ){
		for( size_t batch = 0; batch < bs; ++batch )
			temp[ batch ]= in[ i + batch * num ];
		cipherTexts.push_back( createCipherText( temp ) );
	}
	Shape oldShape = tensor->shape;
	tensor->flatten();
	tensor->init( cipherTexts );
	tensor->reshape( oldShape );
}


void HELibCipherTextFactory::feedCipherTensor( const std::vector<float>& in, TensorP<HELibCipherText> tensor ){
	std::vector<double> doubleVector( in.begin(), in.end() );
	return feedCipherTensor( doubleVector, tensor );
}


void HELibCipherTextFactory::feedCipherTensor( const TensorP<double> in, Tensor<HELibCipherText>& tensor ){
	const uint bs = batchsize();
	// create a vector of ciphertexts
//	std::cout << "batchsize " <<  bs << std::endl;
//	std::cout << "in size " <<  in->shape.capacity() << std::endl;

	const size_t num = in->shape.capacity() / bs; // number of elements in a batch

	// do some sanity checking
	size_t numCheck = 1;
	for( size_t i = 1; i < tensor.shape.size; ++i )
		numCheck *= tensor.shape[ i ];
//	std::cout << num << std::endl;
//	std::cout << numCheck << std::endl;
	if( num != numCheck )
		throw std::logic_error( "Shape does not match supported batchsize" );


	std::vector<double> temp( bs, 0 );
	std::vector<HELibCipherText> cipherTexts;
	for( size_t i=0; i<num; ++i ){
		for( size_t batch = 0; batch < bs; ++batch ){
			temp[ batch ]= in->operator []( i + batch * num);
		}
		cipherTexts.push_back( createCipherText( temp ) );
	}
//	std::cout << std::endl;


	Shape oldShape = tensor.shape;
	tensor.flatten();
	tensor.init( cipherTexts );
	tensor.reshape( oldShape );
}



void HELibCipherTextFactory::feedCipherTensor( const TensorP<double> in, TensorP<HELibCipherText> tensor ){
	feedCipherTensor( in, *tensor );
}

std::ostream& operator<<( std::ostream& output, const HELibCipherText& heCtxt ) {
	// FIXME do something
	output << heCtxt.ctxt().getRatFactor();
	return output;
}














