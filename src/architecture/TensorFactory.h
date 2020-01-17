/*
 * TensorFactory.h
 *
 *  Created on: Mar 5, 2019
 *      Author: robert
 */

#ifndef ARCHITECTURE_TENSORFACTORY_H_
#define ARCHITECTURE_TENSORFACTORY_H_

#include <memory>

class Shape;

/**
 * Abstract base to create Tensors
 */
template<class T>
class TensorFactory {
public:

	virtual TensorP<T> create( Shape s ) = 0;

	/** @brief
	 * Creates a different view on the same datastorage. Shapes must match.
	 * Views can alter the underlying data just as the original Tensor. After
	 * the creation they share equal ownership of the data
	 */
	virtual TensorP<T> createView( Shape s, TensorP<T> other ) = 0;

	virtual ~TensorFactory() {
	}

	//numpy-like funcs
	virtual TensorP<T> onesAndInit( Shape ) = 0;
	virtual TensorP<T> arangeAndInit( Shape, int, int ) = 0;
};

template<class T>
using TensorFactoryP = std::shared_ptr<TensorFactory<T>>;

#endif /* ARCHITECTURE_TENSORFACTORY_H_ */
