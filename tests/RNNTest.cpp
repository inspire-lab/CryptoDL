
/*
#include "../../test/TestCommonsImpl.h"
 * RNNTest.cpp
 *
 *  Created on: Apr 11, 2019
 *      Author: robert
 */

#include "RNNTest.h"
#include "TestCommons.h"
#include "../src/architecture/ActivationFunction.h"
#include "../src/architecture/Layer.h"
#include "../src/architecture/PlainTensor.h"




bool rnnTest1() {

	uint datadim = 3;
	uint timesteps = 2;
	uint units = 2;
	PlainTensorFactory<float> factory;
	auto input = factory.ones( Shape( { 1, timesteps, datadim } ) );

	RNN<float, float, PlainTensor<float>, PlainTensor<float>>
	layer( "rnn", LinearActivation<float>::getSharedPointer(), units, false, input, &factory, &factory );
	layer.output()->init();

	layer.weights( factory.ones( Shape( { units, datadim } ) ) );
	layer.biases( factory.zeros( Shape( { units } ) ) );
	layer.recurrentWeights( factory.ones( Shape( { units, units } ) ) );

	auto expectedOutput = factory.create( { 2 } );
	std::vector<float> tmp { 9, 9 };
	expectedOutput->init( tmp );
	expectedOutput->reshape( Shape { 1, 2 } );

	layer.feedForward();
//	std::cout << *layer.output() << std::endl;
//	std::cout << *expectedOutput << std::endl;
	return finishTest( layer.output(), expectedOutput, __func__ );

}

bool rnnTest2() {

	uint datadim = 3;
	uint timesteps = 3;
	uint units = 2;
	PlainTensorFactory<float> factory;
	auto input = factory.create( Shape( { 1, timesteps, datadim } ) );
	std::vector<std::vector<std::vector<float>>> inv { std::vector<std::vector<float>> { std::vector<float> { 1, 1, 1 }, std::vector<float> { 0, 0, 0 },
			std::vector<float> { 0, 0, 0 } } };
	input->init( inv );

	RNN<float, float, PlainTensor<float>, PlainTensor<float>>
	layer( "rnn", LinearActivation<float>::getSharedPointer(), units, true, input, &factory, &factory );
	layer.output()->init();

	layer.weights( factory.ones( Shape( { units, datadim } ) ) );
	layer.biases( factory.zeros( Shape( { units } ) ) );
	auto rwv = factory.range( Shape( { units, units } ) );
//	std::cout << "recurrentweights: " << std::endl;
//	std::cout << *rwv << std::endl;
	layer.recurrentWeights( rwv );

	auto expectedOutput = factory.create( Shape { 1, 3, 2 } );
	std::vector<std::vector<std::vector<float>>> tmp { std::vector<std::vector<float>> { std::vector<float> { 3, 3 }, std::vector<float> { 3, 15 },
			std::vector<float> { 15, 51 } } };
	expectedOutput->init( tmp );

	layer.feedForward();
//	std::cout << "ours" << std::endl;
//	std::cout << *layer.output() << std::endl;
//	std::cout << "keras" << std::endl;
//	std::cout << *expectedOutput << std::endl;
	return finishTest( layer.output(), expectedOutput, __func__ );

}

bool rnnTest3() {

	uint datadim = 3;
	uint timesteps = 3;
	uint units = 2;
	PlainTensorFactory<float> factory;
	auto input = factory.range( Shape( { 1, timesteps, datadim } ) );

	RNN<float, float, PlainTensor<float>, PlainTensor<float>>
	layer( "rnn", LinearActivation<float>::getSharedPointer(), units, true, input, &factory, &factory );
	layer.output()->init();

	layer.weights( factory.range( Shape( { units, datadim } ) ) );
	layer.biases( factory.zeros( Shape( { units } ) ) );
	auto rwv = factory.zeros( Shape( { units, units } ) );
	layer.recurrentWeights( rwv );

	auto expectedOutput = factory.create( Shape { 1, 3, 2 } );
	std::vector<std::vector<std::vector<float>>> tmp { std::vector<std::vector<float>> { std::vector<float> { 5, 14 }, std::vector<float> { 14, 50 },
			std::vector<float> { 23, 86 } } };
	expectedOutput->init( tmp );

	layer.feedForward();
	return finishTest( layer.output(), expectedOutput, __func__ );

}

