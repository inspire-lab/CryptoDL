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

	HELibCipherText( std::shared_ptr<helib::Ctxt> ctxt, HELibCipherTextFactory* factory ) :
			mFactory( factory ), mCtxt( ctxt ) {

	}

	helib::Ctxt& ctxt() const {
		return *mCtxt;
	}

	virtual HELibCipherText& operator+=( long x );
	virtual HELibCipherText& operator*=( long x );
	virtual HELibCipherText& operator+=( float x );
	virtual HELibCipherText& operator*=( float x );
	virtual HELibCipherText& operator+=( double x );
	virtual HELibCipherText& operator*=( double x );

	virtual HELibCipherText& operator+=( HELibCipherText* other ) {
		// *mCtxt += ( (HELibCipherText*) other )->ctxt();
		mCtxt->multiplyBy( other->ctxt() );
		return *this;
	}

	virtual HELibCipherText& operator*=( HELibCipherText* other ) {
		*mCtxt *= ( (HELibCipherText*) other )->ctxt();
		return *this;
	}

	virtual HELibCipherText& operator+=( HELibCipherText& other );

	virtual HELibCipherText& operator*=( HELibCipherText& other ) {
		// *mCtxt *= ( (HELibCipherText&) other ).ctxt();
		mCtxt->multiplyBy( other.ctxt() );
		return *this;
	}

	HELibCipherText& operator=( const HELibCipherText& other ) {
		*mCtxt = ( (HELibCipherText&) other ).ctxt();
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
	void writeToFile( const std::string& fileName ){
		std::fstream fstr;
		fstr = std::fstream( fileName, std::ios::out | std::ios::binary );
		if ( !fstr ){
			std::cerr << "something messed up when opening " << fileName << std::endl;
			exit( 1 );
		}

		ctxt().write( fstr );
		fstr.close();
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
	std::shared_ptr<helib::Ctxt> mCtxt; 	// needs to wrapped in a pointer because of its = operator
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
 * Parameters for the CKKS constructor
 * HELibCipherTextFactory( long L, long m, long r, long c = 2 ) :
 * const long L;         // Number of bits (kinda like levels)
 * const long m;         // Zm*
 * const long r;         // bit precision
 *
 * Some experimentally determined values for at least 128bit security
 * 
 * L = 50 M = 2^13 r = 16
 * L = 100 M = 2^13 r = 16
 * L = 150 M = 2^14 r = 16
 * L = 200 M = 2^14 r = 16
 * L = 250 M = 2^15 r = 16
 * L = 300 M = 2^15 r = 16
 * L = 350 M = 2^15 r = 16
 * L = 400 M = 2^15 r = 16
 * L = 450 M = 2^15 r = 16
 * L = 500 M = 2^15 r = 16
 * L = 550 M = 2^16 r = 16
 * L = 600 M = 2^16 r = 16
 * L = 650 M = 2^16 r = 16
 * L = 700 M = 2^16 r = 16
 * L = 750 M = 2^16 r = 16
 * L = 800 M = 2^16 r = 16
 * L = 850 M = 2^16 r = 16
 * L = 900 M = 2^16 r = 16
 * L = 950 M = 2^16 r = 16
 * L = 1000 M = 2^16 r = 16
 * L = 1050 M = 2^16 r = 16
 * 
 * L = 50 M = 2^13 r = 32
 * L = 100 M = 2^13 r = 32
 * L = 150 M = 2^14 r = 32
 * L = 200 M = 2^14 r = 32
 * L = 250 M = 2^15 r = 32
 * L = 300 M = 2^15 r = 32
 * L = 350 M = 2^15 r = 32
 * L = 400 M = 2^15 r = 32
 * L = 450 M = 2^15 r = 32
 * L = 500 M = 2^15 r = 32
 * L = 550 M = 2^16 r = 32
 * L = 600 M = 2^16 r = 32
 * L = 650 M = 2^16 r = 32
 * L = 700 M = 2^16 r = 32
 * L = 750 M = 2^16 r = 32
 * L = 800 M = 2^16 r = 32
 * L = 850 M = 2^16 r = 32
 * L = 900 M = 2^16 r = 32
 * L = 950 M = 2^16 r = 32
 * L = 1000 M = 2^16 r = 32
 * L = 1050 M = 2^16 r = 32
 */
class HELibCipherTextFactory: public CipherTextWrapperFactory<HELibCipherText> {
public:

	const bool useBFV;

	HELibCipherTextFactory (const HELibCipherTextFactory& );

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
			context = std::make_shared<helib::Context>( m, p, r );
			buildModChain( *context, L, c ); 				// Modify the context, adding primes to the modulus chain

			secretKey = std::make_shared<helib::SecKey>( *context );
			secretKey->GenSecKey( w );
			helib::addSome1DMatrices( *secretKey );					// compute key-switching matrices that we need
			publicKey = secretKey;
			G = helib::makeIrredPoly( p, d );

			context->ea;
			ea = std::make_shared<helib::EncryptedArray>( *context, G );
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
			context = std::make_shared<helib::Context>( /*m=*/m, /*p=*/-1, r ); /// just using the examples given by HELib docu.
			buildModChain( *context, L, c ); 				// Modify the context, adding primes to the modulus chain
			secretKey = std::make_shared<helib::SecKey>( *context );
			secretKey->GenSecKey();
			helib::addSome1DMatrices( *secretKey );					// compute key-switching matrices that we need
			publicKey = secretKey;
			ea = std::make_shared<helib::EncryptedArray>( *context );
		}

	}

	/**
	* const long m;         // Zm*
 	* const long r;         // bit precision
 	* const long L;         // Number of bits
	*/
	HELibCipherTextFactory( long L, long m, long r, long c = 2 ) :
			useBFV( false ) {

		// SetSeed( NTL::ZZ( 0 ) );
		/// m specific the ring Zm*
		/// p = -1 means CKKS
		/// r is the number of bits after the decimal aka precision 
		/// L Number of bits
		context = std::make_shared<helib::Context>( /*m=*/m, /*p=*/-1, r ); /// just using the examples given by HELib docu.
		helib::buildModChain( *context, L, c ); 				// Modify the context, adding primes to the modulus chain
		secretKey = std::make_shared<helib::SecKey>( *context );
		secretKey->GenSecKey();
		helib::addSome1DMatrices( *secretKey );					// compute key-switching matrices that we need
		// helib::addAllMatrices( *secretKey );
		publicKey = secretKey;
		ea = std::make_shared<helib::EncryptedArray>( *context );
		// std::cout << "security level: " << context->securityLevel() << std::endl;

	}

	HELibCipherTextFactory( const std::string& contextFileName, const std::string& pubkeyFileName  ) :
			useBFV( false ) {

		// read context
		std::fstream fstr;
		fstr = std::fstream( contextFileName, std::ios::in | std::ios::binary );
		if ( !fstr ){
			std::cerr << "something messed up when opening " << contextFileName << std::endl;
			exit( 1 );
		}
		context = helib::buildContextFromBinary( fstr );
		helib::readContextBinary( fstr, *context );
		fstr.close();
		
		
		// read publickey
		fstr = std::fstream( pubkeyFileName, std::ios::in | std::ios::binary );
		if ( !fstr ){
			std::cerr << "something messed up when opening " << pubkeyFileName << std::endl;
			exit( 1 );
		}
		publicKey =  std::make_shared<helib::PubKey>( *context );
		helib::readPubKeyBinary( fstr, *publicKey );
		fstr.close();

		ea = std::make_shared<helib::EncryptedArray>( *context );
		std::cout << "security level: " << context->securityLevel() << std::endl;

		secretKey = NULL;

	}

	HELibCipherTextFactory( const std::string& contextFileName, const std::string& pubkeyFileName, const std::string& secFileName ) :
			HELibCipherTextFactory( contextFileName, pubkeyFileName ) {

		// read secrectkey
		std::fstream fstr;
		fstr = std::fstream( secFileName, std::ios::in | std::ios::binary );
		if ( !fstr ){
			std::cerr << "something messed up when opening " << secFileName << std::endl;
			exit( 1 );
		}
		secretKey = std::make_shared<helib::SecKey>( *context );
		helib::readSecKeyBinary( fstr, *secretKey );
		fstr.close();
	}
	
	virtual HELibCipherText empty() override {
		return HELibCipherText( std::make_shared<helib::Ctxt>( *publicKey ), this );
	}

	virtual HELibCipherText createCipherText( long x ) override {
		std::shared_ptr<helib::Ctxt> ctxt = std::make_shared<helib::Ctxt>( *publicKey );
		if ( useBFV ) {
			publicKey->Encrypt( *ctxt, NTL::to_ZZX( x ) );
		} else {
			const helib::EncryptedArrayCx& ea = context.get()->ea->getCx();
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

	/**
	 * Encrypts the data and sticks it into the ciphertensor.
	 * If batchSize == -1 the batch size is infered from the crypto parameters otherwise it
	 * must not be larger than the batch size infered from the crypto parameters. 
	 */
	virtual void feedCipherTensor( const std::vector<double>& in, TensorP<HELibCipherText> tensor, int batchSize=-1 ) override;

	/**
	 * Encrypts the data and sticks it into the ciphertensor.
	 * If batchSize == -1 the batch size is infered from the crypto parameters otherwise it
	 * must not be larger than the batch size infered from the crypto parameters. 
	 */
	virtual void feedCipherTensor( const std::vector<float>& in, TensorP<HELibCipherText> tensor, int batchSize=-1 ) override;

	virtual void feedCipherTensor( const TensorP<double> in, TensorP<HELibCipherText> tensor ) override;

	virtual void feedCipherTensor( const TensorP<double> in, Tensor<HELibCipherText>& tensor ) override ;

	/**
	 * experimental multi threaded feeding
	 */
	void feedCipherTensorMultiThread(const std::vector<double>& in, TensorP<HELibCipherText> tensor, int batchSize=-1 );

	virtual uint batchsize() override {
		return ea->size();
	}

	std::shared_ptr<helib::Ctxt> createRawEmpty() {
		return std::make_shared<helib::Ctxt>( *publicKey );
	}


	virtual ~HELibCipherTextFactory() {
	}

	void writeToFile( const std::string& contextFile, const std::string& pubKeyFile, const std::string& seckeyFile ){

		// write context
		std::fstream fstr;
		fstr = std::fstream( contextFile, std::ios::out | std::ios::binary );
		if ( !fstr ){
			std::cerr << "something messed up when opening " << contextFile << std::endl;
			exit( 1 );
		}
		helib::writeContextBaseBinary( fstr, *context );
		helib::writeContextBinary( fstr, *context );
		fstr.close();

		// pubkey
		fstr = std::fstream( pubKeyFile, std::ios::out | std::ios::binary );
		if ( !fstr ){
			std::cerr << "something messed up when opening " << pubKeyFile << std::endl;
			exit( 1 );
		}
		helib::writePubKeyBinary( fstr, *publicKey );

		// write seckey if we have one
		if ( secretKey == NULL || seckeyFile.size() == 0 )
			return;

		// pubkey
		fstr = std::fstream( seckeyFile, std::ios::out | std::ios::binary );
		if ( !fstr ){
			std::cerr << "something messed up when opening " << seckeyFile << std::endl;
			exit( 1 );
		}
		helib::writeSecKeyBinary( fstr, *secretKey );


	}

	/**
	*	Read HElib ctxt from a binary file
	*/
	HELibCipherText readCtxtFromFile( const std::string& fileName ){
		std::fstream fstr;
		fstr = std::fstream( fileName, std::ios::in | std::ios::binary );
		if ( !fstr ){
			std::cerr << "something messed up when opening " << fileName << std::endl;
			exit( 1 );
		}
		HELibCipherText ctxt = empty();
		ctxt.ctxt().read( fstr );
		return ctxt;
	}

	double securityLevel(){
		return context->securityLevel();
	}

	std::shared_ptr<helib::SecKey> secretKey = NULL; // FIXME for debugging. should be private
	std::shared_ptr<helib::Context> context;
private:
	std::shared_ptr<helib::EncryptedArray> ea;
	std::shared_ptr<helib::PubKey> publicKey;
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

	virtual uint multiplicativeDepth() {
		if ( a != 0 )
			return 1;
		return 0;
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
	TensorP<HELibCipherText> tensor = nullptr; 

	PolynomialActivationDegree3( float a, float b, float c,float d, TensorP<HELibCipherText> tensor )
			:a( a ), b( b ), c( c ), d( d ),  tensor( tensor ) {
	}
	
		PolynomialActivationDegree3( float a, float b, float c,float d )
			:a( a ), b( b ), c( c ), d( d ){
	}

	void activate( HELibCipherText& in ) {

		if ( tensor == nullptr )
			throw std::runtime_error( "Activation function not properly initialized" );
		/// Need to calculate the invidual parts and then sum it up at the end
		// ax^3
		HELibCipherText ax3 = this-> tensor->empty();
		if( a != 0 ){
			ax3 += in;
			ax3 *= in;
			ax3 *= in;
			ax3 *= a;
		}
		// bx^2
		HELibCipherText bx2 = tensor->empty();
		if( b != 0 ){
			bx2 += in;
			bx2 *= bx2;
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

	virtual uint multiplicativeDepth() {
		if ( a != 0 )
			return 2;
		if ( b != 0 )
			return 1;
		return 0;
	}

	void emptyProvider( TensorP<HELibCipherText> t ) override {
		tensor = t;
	}

	/**
	 * Needs coeffecients a,b,c,d, for ax^3+bx^2+cx+d and TensorP<T> that can provied an empty T.
	 * It does not need initialized storage.
	 */

	static std::shared_ptr<Activation<HELibCipherText>> getSharedPointer( float a, float b, float c,float d, TensorP<HELibCipherText> tensor ) {
		return std::make_shared<PolynomialActivationDegree3<HELibCipherText>>(a,b,c,d,tensor);
	}

		static std::shared_ptr<Activation<HELibCipherText>> getSharedPointer( float a, float b, float c,float d ) {
		return std::make_shared<PolynomialActivationDegree3<HELibCipherText>>(a,b,c,d);
	}

};






#endif /* ARCHITECTURE_HEBACKEND_HELIB_HELIBCIPHERTEXT_H_ */

