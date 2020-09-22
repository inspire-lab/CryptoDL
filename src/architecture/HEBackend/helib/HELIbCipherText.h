/*
 * HELIbCipherText.h
 *
 *  Created on: Mar 3, 2019
 *      Author: robert
 */

#ifndef ARCHITECTURE_HEBACKEND_HELIB_HELIBCIPHERTEXT_H_
#define ARCHITECTURE_HEBACKEND_HELIB_HELIBCIPHERTEXT_H_

#include <memory.h>
#include <NTL/ZZ.h>
#include <helib/FHE.h>
#include <ostream>
#include <helib/EncryptedArray.h>
#include "../CipherTextWrapper.h"
#include "../../ActivationFunction.h"
#include "../HETensor.h"


class HELibCipherTextFactory;


class HELibCipherText {
public:

	friend HELibCipherTextFactory;

	static HELibCipherTextFactory* defaultFactory; // declaration in cpp

	/**
	 * Threshold for the noise. If the noise grows larger
	 * reincryption is triggered if enabled.
	 */
	static long noiseThreshold;

	HELibCipherText();

	HELibCipherText( std::shared_ptr<Ctxt> ctxt, HELibCipherTextFactory* factory ) :
			mFactory( factory ), mCtxt( ctxt ) {

	}

	const Ctxt& ctxt() const {
		return *mCtxt;
	}

	virtual HELibCipherText& operator+=( long x );
	virtual HELibCipherText& operator*=( long x );
	virtual HELibCipherText& operator+=( float x );
	virtual HELibCipherText& operator*=( float x );
	virtual HELibCipherText& operator+=( double x );
	virtual HELibCipherText& operator*=( double x );

	virtual HELibCipherText& operator+=( HELibCipherText* other ) {
		*mCtxt += ( (HELibCipherText*) other )->ctxt();
		return *this;
	}

	virtual HELibCipherText& operator*=( HELibCipherText* other ) {
		*mCtxt *= ( (HELibCipherText*) other )->ctxt();
		return *this;
	}

	virtual HELibCipherText& operator+=( HELibCipherText& other );

	virtual HELibCipherText& operator*=( HELibCipherText& other ) {
		*mCtxt *= ( (HELibCipherText&) other ).ctxt();
		return *this;
	}

	virtual HELibCipherText empty();

	void square(){
		mCtxt->square();
	}

	void power( uint p ){
		mCtxt->power( p );
	}


	// TODO make me pretty
	friend std::ostream& operator<<( std::ostream& output, const HELibCipherText& heCtxt );
	
	/**
	*	Write the HElib ctxt to a binary file
	*/
	void writeToFile( std::ostream& str ){
		ctxt().write( str );
	}

	/**
	 * @brief Checks if the noise is about to overflow
	 */
	bool noiseNearOverflow(){
		//if ( ctxt().getNoiseBound() != 0 )
		//	std::cout << ctxt().getNoiseBound().e << ' ' << noiseThreshold << std::endl;
		return ctxt().getNoiseBound().e >= noiseThreshold;
	}

	virtual ~HELibCipherText() {
	}

	// FIXME move back to private
	HELibCipherTextFactory* mFactory;
	std::shared_ptr<Ctxt> mCtxt; 	// needs to wrapped in a pointer because of its = operator
									// which only allows assignment to ctxt with the same context
									// and that makes things annoying
private:

};

/**
 * Factory to create `HELibCipherText`
 *
 * Supports both BFV and CKKS crypto schemes. BFV is not properly tested
 * and might be removed in the future. The big drawback of BFV is that it
 * does not supoort float numbers
 *
 *
 */
class HELibCipherTextFactory: public CipherTextWrapperFactory<HELibCipherText> {
public:

	const bool useBFV;

	HELibCipherTextFactory( long seed = 0, bool useBFV = true ) :
			useBFV( useBFV ) {

		SetSeed( NTL::ZZ( seed ) );


		if ( useBFV ) {
			long p = 4999; // Plaintext prime modulus
			long L = 300;	/// Number of levels in the modulus chain [default=heuristic]
							/// if used with out the FindM function  it is  Number of bits of the modulus chain
//			long m = 32109;	// Specific modulus
			long m = 2;	// Specific modulus
			long r = 1;		// Lifting [default=1]
			long c = 2;		// Number of columns in key-switching matrix [default=2]
			long d = 1;		// Degree of the field extension [default=1]
			//FIXME: unused variables (remove/use)?
			//long k = 80;	// Security parameter [default=80]
			//long s = 32;	// Minimum number of slots [default=0]
			long w = 64;	// Hamming weight of secret key

			// unsed at the moment
//			long k = 80;	// Security parameter [default=80]
//			long s = 32;	// Minimum number of slots [default=0]

			NTL::ZZX G;
//			m = FindM( k, L, c, p, d, s, 0, true );		// Find a value for m given the specified values
			context = std::make_shared<FHEcontext>( m, p, r );
			buildModChain( *context, L, c ); 				// Modify the context, adding primes to the modulus chain

			secretKey = std::make_shared<FHESecKey>( *context );
			secretKey->GenSecKey( w );
			addSome1DMatrices( *secretKey );					// compute key-switching matrices that we need
			publicKey = secretKey;
			G = makeIrredPoly( p, d );

			context->ea;
			ea = std::make_shared<EncryptedArray>( *context, G );
		}
		else {
			long L = 128;	// Number of levels in the modulus chain [default=heuristic]
			long m = 4096;	/// the ring we work on needs to a power of 2
							/// Defines the batchSize. The batchsize is m/4
			long r = 1;		// Bits of prescion
			long c = 2;		// Number of columns in key-switching matrix [default=2]

			/// m specific the ring
			/// p = -1 means CKKS
			/// r is the number of bits after the decimal aka precision
			context = std::make_shared<FHEcontext>( /*m=*/m, /*p=*/-1, r ); /// just using the examples given by HELib docu.
			buildModChain( *context, L, c ); 				// Modify the context, adding primes to the modulus chain
			secretKey = std::make_shared<FHESecKey>( *context );
			secretKey->GenSecKey();
			addSome1DMatrices( *secretKey );					// compute key-switching matrices that we need
			publicKey = secretKey;
			ea = std::make_shared<EncryptedArray>( *context );
		}

	}

	HELibCipherTextFactory( long L, long m, long r, long c = 2 ) :
			useBFV( false ) {

		SetSeed( NTL::ZZ( 0 ) );
		/// m specific the ring
		/// p = -1 means CKKS
		/// r is the number of bits after the decimal aka precision
		context = std::make_shared<FHEcontext>( /*m=*/m, /*p=*/-1, r ); /// just using the examples given by HELib docu.
		buildModChain( *context, L, c ); 				// Modify the context, adding primes to the modulus chain
		secretKey = std::make_shared<FHESecKey>( *context );
		secretKey->GenSecKey();
		addSome1DMatrices( *secretKey );					// compute key-switching matrices that we need
		publicKey = secretKey;
		ea = std::make_shared<EncryptedArray>( *context );

	}

	virtual HELibCipherText empty() override {
		return HELibCipherText( std::make_shared<Ctxt>( *publicKey ), this );
	}

	virtual HELibCipherText createCipherText( long x ) override {
		std::shared_ptr<Ctxt> ctxt = std::make_shared<Ctxt>( *publicKey );
		if ( useBFV ) {
			publicKey->Encrypt( *ctxt, NTL::to_ZZX( x ) );
		} else {
			EncryptedArrayCx ea = context.get()->ea->getCx();
			std::vector<long> vdLong;
			for ( long i = 0; i < ea.size(); i++ ) {
				vdLong.push_back( x );
			}
			ea.encrypt( *ctxt, *publicKey, vdLong );
		}
		return HELibCipherText( ctxt, this );
	}


	virtual HELibCipherText createCipherText( const std::vector<long> & in ) override;

	virtual HELibCipherText createCipherText( const std::vector<double>& in ) override;

	virtual HELibCipherText createCipherText( const std::vector<float>& in ) override;

	virtual std::vector<long> decryptLong( const HELibCipherText& ctx ) override;

	virtual std::vector<double> decryptDouble( const HELibCipherText& ctx ) override;

	virtual void setAsDefaultFactory() override;

	virtual TensorP<HELibCipherText> createCipherTensor( const std::vector<double>& in, const Shape& shape, HETensorFactory<HELibCipherText>* hetf ) override;

	virtual TensorP<HELibCipherText> createCipherTensor( const std::vector<float>& in, const Shape& shape, HETensorFactory<HELibCipherText>* hetf ) override;

	virtual void feedCipherTensor( const std::vector<double>& in, TensorP<HELibCipherText> tensor ) override;


	virtual void feedCipherTensor( const std::vector<float>& in, TensorP<HELibCipherText> tensor ) override;

	virtual void feedCipherTensor( const TensorP<double> in, TensorP<HELibCipherText> tensor ) override;

	virtual void feedCipherTensor( const TensorP<double> in, Tensor<HELibCipherText>& tensor ) override ;

	virtual uint batchsize() override {
		return ea->size();
	}

	std::shared_ptr<Ctxt> createRawEmpty() {
		return std::make_shared<Ctxt>( *publicKey );
	}


	virtual ~HELibCipherTextFactory() {
	}

	std::shared_ptr<FHESecKey> secretKey; // FIXME for debugging. should be private
private:
	std::shared_ptr<EncryptedArray> ea;
	std::shared_ptr<FHEcontext> context;
	std::shared_ptr<FHEPubKey> publicKey;
};





// TODO move this somewhere nice
/*
 *
 *
 * Template Specification for HElib ciphertexts
 *
 *
 *
 */
/**
 * Degree 2
 * ax^2+bx+c
 *
 */

template<>
class PolynomialActivation<HELibCipherText>: public Activation<HELibCipherText> {

public:
	const float a, b, c;
	const TensorP<HELibCipherText> tensor;
	PolynomialActivation( float a, float b, float c, TensorP<HELibCipherText> tensor )
				:
						a( a ), b( b ), c( c ), tensor( tensor ) {

		}

	void activate( HELibCipherText& in ) {
		/// Need to calculate the invidual parts and then sum it up at the end
		// ax^2
		HELibCipherText ax2 = tensor->empty();
		if( a != 0 ){
			ax2 += in;
			ax2.square();
			ax2 *= a;
		}
		// bx
		HELibCipherText bx = tensor->empty();
		if( b != 0 ){
			bx += in;
			bx *= b;
		}
		// cx
		HELibCipherText result = tensor->empty();
		if( c != 0 )
			result += c;
		// sum it up
		if( a != 0 )
			result += ax2;
		if( b != 0 )
			result += bx;
		in = result;
	}


	/**
	 * Needs coeffecients a,b,c for ax^2+bx+c and TensorP<T> that can provide an empty T.
	 * It does not need initialized storage.
	 */

	static std::shared_ptr<Activation<HELibCipherText>> getSharedPointer( float a, float b, float c, TensorP<HELibCipherText> tensor ) {
		return std::make_shared<PolynomialActivation<HELibCipherText>>(a,b,c,tensor);
	}



};




/**
 * Degree 3
 */
template <>
class PolynomialActivationDegree3<HELibCipherText> : public Activation<HELibCipherText>{

public:

	const float a, b, c, d;
	const TensorP<HELibCipherText> tensor;

	PolynomialActivationDegree3( float a, float b, float c,float d, TensorP<HELibCipherText> tensor )
			:a( a ), b( b ), c( c ), d( d ),  tensor( tensor ) {
	}


	void activate( HELibCipherText& in ) {
		/// Need to calculate the invidual parts and then sum it up at the end
		// ax^3
		HELibCipherText ax3 = this-> tensor->empty();
		if( a != 0 ){
			ax3 += in;
			ax3.power(3);
			ax3 *= a;
		}
		// bx^2
		HELibCipherText bx2 = tensor->empty();
		if( b != 0 ){
			bx2 += in;
			bx2.square();
			bx2 *= b;
		}
		// cx
		HELibCipherText cx = tensor->empty();
		if( c != 0 ){
			cx += in;
			cx *= c;
		}
		// dx
		HELibCipherText result = tensor->empty();
		if( d != 0 )
			result += d;
		// sum it up
		if( a != 0 )
			result += ax3;
		if( b != 0 )
			result += bx2;
		if( c != 0 )
			result += cx;
		in = result;
	}

	/**
	 * Needs coeffecients a,b,c,d, for ax^3+bx^2+cx+d and TensorP<T> that can provied an empty T.
	 * It does not need initialized storage.
	 */

	static std::shared_ptr<Activation<HELibCipherText>> getSharedPointer( float a, float b, float c,float d, TensorP<HELibCipherText> tensor ) {
		return std::make_shared<PolynomialActivationDegree3<HELibCipherText>>(a,b,c,d,tensor);
	}

};






#endif /* ARCHITECTURE_HEBACKEND_HELIB_HELIBCIPHERTEXT_H_ */

