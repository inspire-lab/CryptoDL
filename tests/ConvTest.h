/*
 * ConvTest.h
 *
 *  Created on: Jan 23, 2019
 *      Author: robert
 */

#ifndef TEST_CONVTEST_H_
#define TEST_CONVTEST_H_

#include <gmpxx.h>
#include <errno.h>
#include "../src/architecture/LoadModelData.h"
#include "../src/data/DatasetOperations.h"
#include "TestCommons.h"
#include "../src/architecture/ActivationFunction.h"
#include "../src/architecture/Layer.h"
#include "../src/architecture/PlainTensor.h"
#include "TestCommons.h"

extern std::vector<std::vector<std::vector<long>>> loadConvLayerOutput( std::string filename, int batchSize, int depth,
		int size, bool supressOutput = false );


// same padding
template<class T>
bool executeConvTestSame( std::string funcName,
		std::vector<std::vector<std::vector<std::vector<T>>>>& inputVolume,
		std::vector<std::vector<std::vector<std::vector<T>>>>& weightsV, std::vector<T>& biasV,
		std::vector<std::vector<std::vector<std::vector<T>>>>& expectedOutputV ) {
	try {
		PlainTensorFactory<T> factory;
		TensorP<T> input = factory.create( Shape( { 1, 1, 4, 4 } ) );

		input->init( inputVolume );

		TensorP<T> weights = factory.create( Shape( { 1, 1, 3, 3 } ) );
		weights->init( weightsV );
		TensorP<T> biases = factory.create( Shape( { 1 } ) );
		biases->init( biasV );

		TensorFactoryP<T> sharedFactory =
				std::make_shared<PlainTensorFactory<T>>( factory );

		Convolution2D<T, T, PlainTensor<T>, PlainTensor<T> > layer( "test",
				SquareActivation<T>::getSharedPointer(), 1,
				3, 1,
				PADDING_MODE::SAME, input, sharedFactory, sharedFactory );
		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );

		//activate layer
		layer.feedForward();

		//compare w/ expected
		//create expected tensor
		TensorP<T> expectedOutput = factory.create( Shape( { 1, 1, 4, 4 } ) );
		expectedOutput->init( expectedOutputV );
		return finishTest( layer.output(), expectedOutput, funcName );
	} catch ( const std::bad_alloc& e ) {
		std::cout << "Allocation failed: " << e.what() << std::endl;
		return false;
	}
}

bool convTest1_samePad();

bool convTest2_samePad();

bool convTest3_samePad();

bool convTest4_samePad();

bool convTest5_samePad();

bool convTest6_samePad();

bool convTest7_samePad();

bool convTest_samePad_evenKernel1();

bool convTest_samePad_evenKernel2();

bool convTest_strides_samePadding_1();

bool convTest_multiple_channels_non_unit_stride_same_padding_1();

bool convTest_multiple_filters__multiple_channels_non_unit_stride_same_padding_1();

bool convTest_batches_multiple_filters__multiple_channels_non_unit_stride_same_padding_1();

/*
 * uses in and output from a real float network
 */
bool convTest1_samePad_floats();

bool convTest2_samePad_floats();

//extra tests
bool convRangeSameEvenTest1();

bool convRangeSameOddTest1();

bool convMultipleFiltersRangeSameEvenTest1();

bool convMultipleFiltersRangeSameOddTest1();

bool convMultipleChannelsRangeSameEvenTest1();

bool convMultipleChannelsRangeSameOddTest1();

bool convMultipleChannelsMultipleFiltersRangeSameEvenTest1();

bool convMultipleChannelsMultipleFiltersRangeSameOddTest1();

// valid padding
template<class T>
bool executeConvTestValid( std::string funcName,
		std::vector<std::vector<std::vector<std::vector<T>>>>& inputVolume,
		std::vector<std::vector<std::vector<std::vector<T>>>>& weightsV, std::vector<T>& biasV,
		std::vector<std::vector<std::vector<std::vector<T>>>>& expectedOutputV ) {
	try {
		PlainTensorFactory<T> factory;
		TensorP<T> input = factory.create( Shape( { 1, 1, 4, 4 } ) );

		input->init( inputVolume );

		TensorP<T> weights = factory.create( Shape( { 1, 1, 3, 3 } ) );
		weights->init( weightsV );
		TensorP<T> biases = factory.create( Shape( { 1 } ) );
		biases->init( biasV );

		Convolution2D<T, T, PlainTensor<T>, PlainTensor<T>> layer( "test", SquareActivation<T>::getSharedPointer(), 1,
				3, 1, PADDING_MODE::VALID, input, &factory, &factory );
		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );
		//activate layer
		layer.feedForward();

		//compare w/ expected
		//create expected tensor
		TensorP<T> expectedOutput = factory.create( Shape( { 1, 1, 2, 2 } ) );
		expectedOutput->init( expectedOutputV );
		return finishTest( layer.output(), expectedOutput, __func__ );
	} catch ( const std::bad_alloc& e ) {
		std::cout << "Allocation failed: " << e.what() << std::endl;
		return false;
	}
}

bool convTest1_validPad();

bool convTest2_validPad();

bool convTest3_validPad();

bool convTest4_validPad();

bool convTest5_validPad();

bool convTest_strides_validPad_1();

bool convTest_multiple_channels_non_unit_stride_validPad_1();

bool convTest_multiple_filters__multiple_channels_non_unit_stride_validPad_1();

bool convTest_batches_multiple_filters__multiple_channels_non_unit_stride_validPad_1();

bool convTest1_validPad_evenfilterszie();

// load output from layer 1 and feed it to layer2
bool convTest_Layer2();

bool flattenImages();

bool convTest_validPad_secondLayer_cryptonet();

#endif /* TEST_CONVTEST_H_ */
