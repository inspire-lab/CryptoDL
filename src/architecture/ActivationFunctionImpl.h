/*
 * ActivationsFunctionImpl.h
 *
 *  Created on: Feb 20, 2019
 *      Author: robert
 *
 *
 *
 * Contains the implementation of activation functions.
 *
 * Note that all activations need to work in place.
 *
 *
 */

#ifndef ARCHITECTURE_ACTIVATIONFUNCTIONIMPL_H_
#define ARCHITECTURE_ACTIVATIONSFUNCTIONIMPL_H_

#include <memory>
#include "Tensor.h"


template<class T>
class Activation {
public:

	virtual void activate( T& in ) = 0;


	static std::shared_ptr<Activation<T>> getSharedPointer() {
		return nullptr;
	}
	;

	//can prob force to be pure abstract class but need to also define destructors in child classes
	virtual ~Activation() {
	}
};

template<class T>
class LinearActivation: public Activation<T> {

	void activate( T& in ) override {
		// linear is super easy. we just do nothing
	}



public:
	static std::shared_ptr<Activation<T>> getSharedPointer() {
		return std::make_shared<LinearActivation<T>>();
	}

};

template<class T>
class SquareActivation: public Activation<T> {

	void activate( T& in ) override {
		in *= in;
	}


public:
	static std::shared_ptr<Activation<T>> getSharedPointer() {
		return std::make_shared<SquareActivation<T>>();
	}

};

/**
 *
 * ax^2+bx+c
 *
 */

template<class T>
class PolynomialActivation: public Activation<T> {

public:
	const float a, b, c;
	const TensorP<T> tensor;
	PolynomialActivation( float a, float b, float c, TensorP<T> tensor )
				:
						a( a ), b( b ), c( c ), tensor( tensor ) {
		}

	void activate( T& in ) override {
		/// Need to calculate the invidual parts and then sum it up at the end
		// ax^2
		T ax2 = tensor->empty();
		if( a != 0 ){
			ax2 += in;
			ax2 *= in;
			ax2 *= a;
		}
		// bx
		T bx = tensor->empty();
		if( b != 0 ){
			bx += in;
			bx *= b;
		}
		// c
		T result = tensor->empty();
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
	 * Needs coeffecients a,b,c for ax^2+bx+c and TensorP<T> that can provied an empty T.
	 * It does not need initialized storage.
	 */

	static std::shared_ptr<Activation<T>> getSharedPointer( float a, float b, float c, TensorP<T> tensor ) {
		return std::make_shared<PolynomialActivation<T>>(a,b,c,tensor);
	}



};

/**
 *
<<<<<<< HEAD
 * ax^4+bx^3+cx^2+dx+e
=======
 * ax^3+bx^2+cx+d
>>>>>>> rnn
 *
 */

template<class T>
class PolynomialActivationDegree3: public Activation<T> {

public:
	const float a, b, c, d;
	const TensorP<T> tensor;

	PolynomialActivationDegree3( float a, float b, float c,float d, TensorP<T> tensor )
			:a( a ), b( b ), c( c ), d( d ),  tensor( tensor ) {
	}

	void activate( T& in ) override {
		/// Need to calculate the individual parts and then sum it up at the end
		// ax^3
		T ax3 = tensor->empty();
		if( a != 0 ){
			ax3 += in;
			ax3 *= in;
			ax3 *= in;
			ax3 *= a;
		}
		// bx^2
		T bx2 = tensor->empty();
		if( b != 0 ){
			bx2 += in;
			bx2 *= in;
			bx2 *= b;
		}
		// cx
		T cx = tensor->empty();
		if( c != 0 ){
			cx += in;
			cx *= c;
		}
		// dx
		T result = tensor->empty();
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

	static std::shared_ptr<Activation<T>> getSharedPointer( float a, float b, float c,float d, TensorP<T> tensor ) {
		return std::make_shared<PolynomialActivationDegree3<T>>(a,b,c,d,tensor);
	}


};

// FIXME still broken. look at the polynomials on how to fix it.
/**
 *
 * ax^4+bx^3+cx^2+dx+e
 *
 */

template<class T>
class PolynomialActivationDegree4: public Activation<T> {

public:
	const float a, b, c, d, e;
	const TensorP<T> tensor;

	PolynomialActivationDegree4( float a, float b, float c,float d,float e, TensorP<T> tensor )
			:a( a ), b( b ), c( c ), d( d ), e( e ), tensor( tensor ) {
	}

	void activate( T& in ) override {
		/// Need to calculate the invidual parts and then sum it up at the end
		// ax^4
		T ax4 = tensor->empty();
		if( a != 0 ){
			ax4 += in;
			ax4 *= in;
			ax4 *= in;
			ax4 *= in;
			ax4 *= a;
		}
		// bx^3
		T bx3 = tensor->empty();
		if( b != 0 ){
			bx3 += in;
			bx3 *= in;
			bx3 *= in;
			bx3 *= b;
		}
		// cx^2
		T cx2 = tensor->empty();
		if( c != 0 ){
			cx2 += in;
			cx2 *= in;
			cx2 *= c;
		}
		// dx
		if( d != 0 )
			in *= d;
		// + e
		if ( e != 0 )
			in += e;
		// sum it up
		if( a != 0 )
			in += ax4;
		if( b != 0 )
			in += bx3;
		if( c != 0 )
			in += cx2;
	}

	/**
	 * Needs coeffecients a,b,c,d,e for x+c ax^4+bx^3+cx^2+dx+e and TensorP<T> that can provied an empty T.
	 * It does not need initialized storage.
	 */

	static std::shared_ptr<Activation<T>> getSharedPointer( float a, float b, float c,float d,float e, TensorP<T> tensor ) {
		return std::make_shared<PolynomialActivationDegree4<T>>(a,b,c,d,e,tensor);
		return std::make_shared<SquareActivation<T>>();
	}

};


template<class T>
class ReluActivation: public Activation<T> {
private:
	T zero = 0;

public:
	static std::shared_ptr<Activation<T>> getSharedPointer() {
		return std::make_shared<ReluActivation<T>>();
	}
	void activate( T& in ) override {
		in = std::max( in, zero );
	}

};

template<class T>
using ActivationP = std::shared_ptr<Activation<T>>;



#endif /* ARCHITECTURE_ACTIVATIONFUNCTIONIMPL_H_ */
