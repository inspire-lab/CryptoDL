
/*
 * DenseTest.cpp
 *
 *  Created on: Feb 6, 2019
 *      Author: robert
 */


#include <vector>
#include "DenseTest.h"
#include "TestCommons.h"
#include "../src/architecture/PlainTensor.h"
#include "../src/architecture/Layer.h"

bool flattenTest1(){
	std::cout << "Running " << __func__ << " " << std::flush;
	std::vector<std::vector<std::vector<long>>> inputVolume { std::vector<std::vector<long>> { std::vector<long> { 1, 2,
			3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 } } };
	std::vector<std::vector<long>> expectedOutputV { std::vector<long> { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
			15, 16 } };
	std::vector<std::vector<long>> flat;

	PlainTensorFactory<long> factory;

	TensorP<long> input = factory.create( Shape( { 1, 1, 16 } ) );
	input->init( inputVolume );

	Flatten<long, long, PlainTensor<long>, PlainTensor<long>> layer( "test", input, &factory, &factory );
	layer.feedForward();

	//compare w/ expected
	//create expected tensor
	TensorP<long> expectedOutput = factory.create( Shape( { 1, 16 } ) );
	expectedOutput->init( expectedOutputV );
	return finishTest( layer.output(), expectedOutput, __func__ );

}


bool flattenTest2(){
	std::cout << "Running " << __func__ << " " << std::flush;
	std::vector<std::vector<std::vector<long>>> inputVolume { std::vector<std::vector<long>> { std::vector<long> { 1, 2,
			3, 4, 5, 6, 7, 8 }, std::vector<
			long> { 9, 10, 11, 12, 13, 14, 15, 16 } } };
	std::vector<std::vector<long>> expectedOutputV { std::vector<long> { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
			15, 16 } };
	std::vector<std::vector<long>> flat;

	PlainTensorFactory<long> factory;
	TensorP<long> input = factory.create( Shape( { 1, 2, 8 } ) );
	input->init( inputVolume );

	Flatten<long, long, PlainTensor<long>, PlainTensor<long>> layer( "test", &factory, &factory );

	layer.input( input );
	layer.buildsOwnOutputTensor();

	layer.feedForward();

	//compare w/ expected
	//create expected tensor
	TensorP<long> expectedOutput = factory.create( Shape( { 1, 16 } ) );
	expectedOutput->init( expectedOutputV );
	return finishTest( layer.output(), expectedOutput, __func__ );

}

bool flattenTest3() {
	std::cout << "Running " << __func__ << " " << std::flush;
	std::vector<std::vector<std::vector<long>>> inputVolume( 2, std::vector<std::vector<long>> { std::vector<long> { 1,
			2, 3, 4, 5, 6, 7, 8 }, std::vector<long> { 9, 10, 11, 12, 13, 14, 15, 16 } } );
	std::vector<std::vector<long>> expectedOutputV( 2, std::vector<long> { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
			14, 15, 16 } );
	std::vector<std::vector<long>> flat;

	PlainTensorFactory<long> factory;
	TensorP<long> input = factory.create( Shape( { 2, 2, 8 } ) );
	input->init( inputVolume );

	Flatten<long, long, PlainTensor<long>, PlainTensor<long>> layer( "test", &factory, &factory );

	layer.input( input );
	layer.buildsOwnOutputTensor();

	layer.feedForward();

	//compare w/ expected
	//create expected tensor
	TensorP<long> expectedOutput = factory.create( Shape( { 2, 16 } ) );
	expectedOutput->init( expectedOutputV );
	return finishTest( layer.output(), expectedOutput, __func__ );

}

bool denseTest1(){
	std::cout << "Running " << __func__ << " " << std::flush;
	std::vector<std::vector<long>> inputVolume { std::vector<long> { std::vector<long>( 6, 1 ) } };
	std::vector<std::vector<long>> weightsV( 3, std::vector<long>( 6, 1 ) );
	std::vector<long> biasV( 3, 1 );
	std::vector<std::vector<long>> expectedOutputV { std::vector<long>( 3, 49 ) };

	try {
		PlainTensorFactory<long> ptFactory;

		TensorP<long> input = ptFactory.create( Shape( { 1, 6 } ) );
		input->init( inputVolume );

		TensorP<long> weights = ptFactory.create( Shape( { 3, 6 } ) );

		weights->init( weightsV );
		TensorP<long> biases = ptFactory.create( Shape( { 3 } ) );
		biases->init( biasV );

		Dense<long, long, PlainTensor<long>, PlainTensor<long> > layer( "test",
				SquareActivation<long>::getSharedPointer(), 3, input, &ptFactory, &ptFactory );

		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );
		//activate layer
		layer.feedForward();

		//compare w/ expected
		//create expected tensor
		TensorP<long> expectedOutput = ptFactory.create( Shape( { 1, 3 } ) );
		expectedOutput->init( expectedOutputV );
		return finishTest( layer.output(), expectedOutput, __func__ );

	} catch ( const std::bad_alloc& e ) {
		std::cout << "Allocation failed: " << e.what() << std::endl;
		return false;
	}
}

bool flattenTest4(){ //FIXME currently incomplete
	std::cout << "Running " << __func__ << " " << std::endl;
//	std::vector<std::vector<std::vector<long>>> inputVolume { std::vector<std::vector<long>> { std::vector<long> { 1, 2,
//			3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 } } };
//	std::vector<std::vector<long>> expectedOutputV { std::vector<long> { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
//			15, 16 } };
//	std::vector<std::vector<long>> flat;

	PlainTensorFactory<long> factory;
	auto input = factory.range( Shape{ 1,3,4,4 } );

	Flatten<long, long, PlainTensor<long>, PlainTensor<long>> layer( "test", input, &factory, &factory, true );
	layer.output()->init();
	layer.feedForward();

	//compare w/ expected
	//create expected tensor
	TensorP<long> expectedOutput = factory.create( Shape( { 1, 16 } ) );

	layer.output()->reshape( Shape{ 1,4,4,3 } );

	std::cout << *layer.output() << std::endl;

//	return finishTest( layer.output(), expectedOutput, __func__ );
	return false;

}


bool denseTest2(){
	std::cout << "Running " << __func__ << " " << std::flush;
	std::vector<std::vector<long>> inputVolume( 2, std::vector<long> {
			std::vector<long>( 6, 1 ) } );
	std::vector<std::vector<long>> weightsV( 3, std::vector<long>( 6, 1 ) );
	std::vector<long> biasV( 3, 1 );
	std::vector<std::vector<long>> expectedOutputV( 2,
			std::vector<long>( 3, 49 ) );

	try {
		PlainTensorFactory<long> ptFactory;

		TensorP<long> input = ptFactory.create( Shape( { 2, 6 } ) );
		input->init( inputVolume );

		TensorP<long> weights = ptFactory.create( Shape( { 3, 6 } ) );

		weights->init( weightsV );
		TensorP<long> biases = ptFactory.create( Shape( { 3 } ) );
		biases->init( biasV );

		Dense<long, long, PlainTensor<long>, PlainTensor<long> > layer( "test",
				SquareActivation<long>::getSharedPointer(), 3, input,
				&ptFactory, &ptFactory );

		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );
		//activate layer
		layer.feedForward();

		//compare w/ expected
		//create expected tensor
		TensorP<long> expectedOutput = ptFactory.create( Shape( { 2, 3 } ) );
		expectedOutput->init( expectedOutputV );
		return finishTest( layer.output(), expectedOutput, __func__ );

	} catch ( const std::bad_alloc& e ) {
		std::cout << "Allocation failed: " << e.what() << std::endl;
		return false;
	}
}

//is implemented if mpf_class can work with abs()
//bool denseTest3() {
/*std::cout << "Running " << __func__ << " " << std::flush;

	std::vector<std::vector<mpf_class>> inputVolume( 2, std::vector<long> {
			std::vector<mpf_class>( 6000, 10000 ) } );
	std::vector<std::vector<mpf_class>> weightsV( 10,
			std::vector<mpf_class>( 6000, 10000 ) );
	std::vector<mpf_class> biasV( 10, 10000 );
	std::vector<std::vector<mpf_class>> expectedOutputV( 2,
			std::vector<mpf_class>( 10,
					mpf_class( "359998298661607209697280" ) ) );

	try {
		PlainTensorFactory<mpf_class> ptFactory;

 TensorP<mpf_class> input = ptFactory.create( Shape( { 2, 6000 } ) );
		input->init( inputVolume );

 TensorP<mpf_class> weights = ptFactory.create( Shape( { 10, 6000 } ) );

		weights->init( weightsV );
 TensorP<mpf_class> biases = ptFactory.create( Shape( { 10 } ) );
		biases->init( biasV );

		Dense<mpf_class, mpf_class, PlainTensor<mpf_class>,
				PlainTensor<mpf_class> > layer( "test",
				SquareActivation<mpf_class>::getSharedPointer(), 3, input,
				&ptFactory, &ptFactory );

		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );
		//activate layer
		layer.feedForward();

		//compare w/ expected
		//create expected tensor
		TensorP<mpf_class> expectedOutput = ptFactory.create(
 Shape( { 2, 10 } ) );
		expectedOutput->init( expectedOutputV );
		return finishTest( layer.output(), expectedOutput, __func__ );

	} catch ( const std::bad_alloc& e ) {
		std::cout << "Allocation failed: " << e.what() << std::endl;
		return false;
 }*/
//	std::vector<std::vector<mpf_class>> in( 2, std::vector<mpf_class>( 6000, 10000 ) );
//
//	std::vector<std::vector<int>> weights( 10, std::vector<int>( 6000, 10000) );
//	std::vector<int> biases( 10, 10000);
//
//	std::vector<std::vector<mpf_class>> reference_out( 2, std::vector<mpf_class>(10, mpf_class( "359998298661607209697280" ) ) );
//
//	std::vector<std::vector<mpf_class>> out;
//	fullyConnectedParallel<mpf_class, int>( in, weights, biases, true, out );
//	std::cout <<  out[0][0] - reference_out[0][0] << endl;
//	return finishTest<mpf_class>( out, reference_out, __func__  );
//	return false;
//}


