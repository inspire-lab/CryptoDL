/*
 * ConvTestSamePadding.cpp
 *
 *  Created on: Jan 23, 2019
 *      Author: robert
 */

#include "ConvTest.h"

using namespace std;

bool convTest1_samePad(){

	cout << "Running " << __func__ << " " << flush;

	vector<vector<vector<vector<long>>>> inputVolume { vector<vector<vector<long>>> { vector<vector<long>>( 4, vector<long> { 1, 1, 1, 1 } ) } };

	vector<vector<vector<vector<long>>>> weightsV { vector<vector<vector<long>>> { vector<vector<long>>( 3, vector<long> { 1, 1, 1 } ) } }; //
	vector<long> biasV { 1 };
	vector<vector<vector<vector<long>>>> expectedOutputV { vector<vector<vector<long>>> { vector<vector<long>> {
			vector<long> { 25, 49, 49, 25 },
			vector<long> { 49, 100, 100, 49 },
			vector<long> { 49, 100, 100, 49 },
			vector<long> { 25, 49, 49, 25 } } } };

//	[[[[ 25.  49.  49.  25.]
//	   [ 49. 100. 100.  49.]
//	   [ 49. 100. 100.  49.]
//	   [ 25.  49.  49.  25.]]]]
	return executeConvTestSame<long>( __func__, inputVolume, weightsV, biasV,
			expectedOutputV );
}

bool convTest2_samePad(){
	cout << "Running " << __func__ << " " << flush;

	vector<vector<vector<vector<long>>>> inputVolume { vector<
			vector<vector<long>>> { vector<vector<long>>( 4, vector<long> { 1,
			2, 2, 1 } ) } };
	vector<vector<vector<vector<long>>>> weightsV {
			vector<vector<vector<long>>> { vector<vector<long>>( 3,
					vector<long> { 1, 1, 1 } ) } }; //
	vector<long> biasV { 1 };
	vector<vector<vector<vector<long>>>> expectedOutputV { vector<
			vector<vector<long>>> { vector<vector<long>> { vector<long> { 49,
			121, 121, 49 }, vector<long> { 100, 256, 256, 100 }, vector<long> {
			100, 256, 256, 100 }, vector<long> { 49, 121, 121, 49 } } } };
//  [[[[ 49. 121. 121.  49.]
//     [100. 256. 256. 100.]
//     [100. 256. 256. 100.]
//     [ 49. 121. 121.  49.]]]]
	return executeConvTestSame<long>( __func__, inputVolume, weightsV, biasV,
			expectedOutputV );
}


bool convTest3_samePad(){
	cout << "Running " << __func__ << " " << flush;

	vector<vector<vector<vector<long>>>> inputVolume { vector<
			vector<vector<long>>> { vector<vector<long>> { vector<long> { 2, 2,
			3, 2 }, vector<long> { 2, 1, 2, 3 }, vector<long> { 1, 3, 2, 3 },
			vector<long> { 2, 2, 1, 2 } } } };
	vector<vector<vector<vector<long>>>> weightsV {
			vector<vector<vector<long>>> { vector<vector<long>>( 3,
					vector<long> { 1, 1, 1 } ) } }; //
	vector<long> biasV { 1 };
	vector<vector<vector<vector<long>>>> expectedOutputV { vector<
			vector<vector<long>>> { vector<vector<long>> { vector<long> { 64,
			169, 196, 121 }, vector<long> { 144, 361, 484, 256 }, vector<long> {
			144, 289, 400, 196 }, vector<long> { 81, 144, 196, 81 } } } };
//  [[[[ 64. 169. 196. 121.]
//     [144. 361. 484. 256.]
//     [144. 289. 400. 196.]
//     [ 81. 144. 196.  81.]]]]
	return executeConvTestSame<long>( __func__, inputVolume, weightsV, biasV,
			expectedOutputV );
}


bool convTest4_samePad(){
	cout << "Running " << __func__ << " " << flush;

	vector<vector<vector<vector<long>>>> inputVolume { vector<
			vector<vector<long>>> { vector<vector<long>> { vector<long> { 1, 3,
			3, 2 }, vector<long> { 2, 1, 3, 2 }, vector<long> { 1, 3, 2, 1 },
			vector<long> { 3, 2, 3, 1 } } } };
	vector<vector<vector<vector<long>>>> weightsV {
			vector<vector<vector<long>>> { vector<vector<long>>( 3,
					vector<long> { 1, 1, 1 } ) } }; //
	vector<long> biasV { 1 };
	vector<vector<vector<vector<long>>>> expectedOutputV { vector<
			vector<vector<long>>> { vector<vector<long>> { vector<long> { 64,
			196, 225, 121 }, vector<long> { 144, 400, 441, 196 }, vector<long> {
			169, 441, 361, 169 }, vector<long> { 100, 225, 169, 64 } } } };
//  [[[[ 64. 196. 225. 121.]
//     [144. 400. 441. 196.]
//     [169. 441. 361. 169.]
//     [100. 225. 169.  64.]]]]
	return executeConvTestSame<long>( __func__, inputVolume, weightsV, biasV,
			expectedOutputV );
}


bool convTest5_samePad(){
	cout << "Running " << __func__ << " " << flush;

	vector<vector<vector<vector<long>>>> inputVolume { vector<
			vector<vector<long>>> { vector<vector<long>> { vector<long> { 3, 2,
			3, 2 }, vector<long> { 2, 4, 1, 3 }, vector<long> { 2, 2, 1, 3 },
			vector<long> { 1, 3, 2, 3 } } } };
	vector<vector<vector<vector<long>>>> weightsV {
			vector<vector<vector<long>>> { vector<vector<long>>( 3,
					vector<long> { 1, 1, 1 } ) } }; //
	vector<long> biasV { 1 };
	vector<vector<vector<vector<long>>>> expectedOutputV { vector<
			vector<vector<long>>> { vector<vector<long>> { vector<long> { 144,
			256, 256, 100 }, vector<long> { 256, 441, 484, 196 }, vector<long> {
			225, 361, 529, 196 }, vector<long> { 81, 144, 225, 100 } } } };
//  [[[[144. 256. 256. 100.]
//     [256. 441. 484. 196.]
//     [225. 361. 529. 196.]
//     [ 81. 144. 225. 100.]]]]
	return executeConvTestSame<long>( __func__, inputVolume, weightsV, biasV,
			expectedOutputV );
}

bool convTest6_samePad(){
	cout << "Running " << __func__ << " " << endl;

	vector<vector<vector<vector<long>>>> inputVolume { vector<
			vector<vector<long>>> { vector<vector<long>> { vector<long> { 1, 2,
			3, 4 }, vector<long> { 5, 6, 7, 8 }, vector<long> { 9, 10, 11, 12 },
			vector<long> { 13, 14, 15, 16 } } } };
	vector<vector<vector<vector<long>>>> weightsV {
			vector<vector<vector<long>>> { vector<vector<long>> { vector<long> {
					1, 2, 3 }, vector<long> { 4, 5, 6 },
					vector<long> { 7, 8, 9 } } } };

	vector<long> biasV { 1 };
	vector<vector<vector<vector<long>>>> expectedOutputV { vector<
			vector<vector<long>>> { vector<vector<long>> { vector<long> { 12544,
			32041, 47524, 21316 },
			vector<long> { 53824, 121801, 155236, 64009 }, vector<long> {
					132496, 279841, 329476, 130321 }, vector<long> { 39204,
					75625, 87616, 30976 } } } };

//	[[[[ 12544.  32041.  47524.  21316.]
//	   [ 53824. 121801. 155236.  64009.]
//	   [132496. 279841. 329476. 130321.]
//	   [ 39204.  75625.  87616.  30976.]]]]

	return executeConvTestSame<long>( __func__, inputVolume, weightsV, biasV,
			expectedOutputV );
}

bool convTest7_samePad(){

	cout << "Running " << __func__ << "  " << flush;
	cout << setprecision(std::numeric_limits<long double>::digits10 + 1);

	float x =0.001f;

	vector<vector<vector<vector<float>>>> inputVolume { vector<
			vector<vector<float>>> { vector<vector<float>> { vector<float> { x,
			x, x, x }, vector<float> { x, x, x, x },
			vector<float> { x, x, x, x }, vector<float> { x, x, x, x } } } };
	vector<vector<vector<vector<float>>>> weightsV { vector<
			vector<vector<float>>> { vector<vector<float>> { vector<float> { x,
			x, x }, vector<float> { x, x, x }, vector<float> { x, x, x } } } }; //
	vector<float> biasV { 1 };
	vector<vector<vector<vector<float>>>> expectedOutputV { vector<
			vector<vector<float>>> { vector<vector<float>> { vector<float> {
			1.0000081062316895, 1.000011920928955, 1.000011920928955,
			1.0000081062316895 }, vector<float> { 1.000011920928955,
			1.0000178813934326, 1.0000178813934326, 1.000011920928955 }, vector<
			float> { 1.000011920928955, 1.0000178813934326, 1.0000178813934326,
			1.000011920928955 }, vector<float> { 1.0000081062316895,
			1.000011920928955, 1.000011920928955, 1.0000081062316895 } } } };

//	[[[[1. 1. 1. 1.]
//	   [1. 1. 1. 1.]
//	   [1. 1. 1. 1.]
//	   [1. 1. 1. 1.]]]]
	return executeConvTestSame<float>( __func__, inputVolume, weightsV, biasV, expectedOutputV );
}

bool convTest_samePad_evenKernel1() {
	cout << "Running " << __func__ << std::endl;

	vector<vector<vector<vector<float>>>> inputVolume = vector<
			vector<vector<vector<float>>>> { vector<vector<vector<float>>> {
			vector<vector<float>> { vector<float> { 0, 1, 2, 3, 4 }, vector<
					float> { 5, 6, 7, 8, 9 },
					vector<float> { 10, 11, 12, 13, 14 }, vector<float> { 15,
							16, 17, 18, 19 },
					vector<float> { 20, 21, 22, 23, 24 } } } };
	vector<vector<vector<vector<float>>>> weightsV = vector<
			vector<vector<vector<float>>>> { vector<vector<vector<float>>> {
			vector<vector<float>> {
					vector<float> { 1, 1 },
					vector<float> { 1, 1 } } } };
	vector<float> biasV { 0 };
	vector<vector<vector<vector<float>>>> expectedOutputV = vector<
			vector<vector<vector<float>>>> { vector<vector<vector<float>>> {
			vector<vector<float>> { vector<float> { 12, 16, 20, 24, 13 },
					vector<float> { 32, 36, 40, 44, 23 }, vector<float> { 52,
							56, 60, 64, 33 },
					vector<float> { 72, 76, 80, 84, 43 }, vector<float> { 41,
							43, 45, 47, 24 } } } };
//	[[[[ 12.  16.  20.  24.  13.]
//	   [ 32.  36.  40.  44.  23.]
//	   [ 52.  56.  60.  64.  33.]
//	   [ 72.  76.  80.  84.  43.]
//	   [ 41.  43.  45.  47.  24.]]]]
	try {
		PlainTensorFactory<float> factory;
		TensorP<float> input = factory.create( Shape( { 1, 1, 5, 5 } ) );

		input->init( inputVolume );

		TensorP<float> weights = factory.create( Shape( { 1, 1, 2, 2 } ) );
		weights->init( weightsV );

		TensorP<float> biases = factory.create( Shape( { 1 } ) );
		biases->init( biasV );

		TensorFactoryP<float> sharedFactory = std::make_shared<
				PlainTensorFactory<float>>( factory );

		Convolution2D<float, float, PlainTensor<float>, PlainTensor<float> > layer(
				"test", LinearActivation<float>::getSharedPointer(), 1, 2, 1,
				PADDING_MODE::SAME, input, sharedFactory, sharedFactory );
		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );
		//activate layer
		layer.feedForward();

		//compare w/ expected
		//create expected tensor
		TensorP<float> expectedOutput = factory.create(
				Shape( { 1, 1, 5, 5 } ) );
		expectedOutput->init( expectedOutputV );
		return finishTest( layer.output(), expectedOutput, __func__ );
	} catch ( const std::bad_alloc& e ) {
		std::cout << "Allocation failed: " << e.what() << std::endl;
		return false;
	}
}

//4x4
bool convTest_samePad_evenKernel2() {
	cout << "Running " << __func__ << std::endl;

	vector<float> biasV { 0 };
	vector<vector<vector<vector<float>>>> expectedOutputV = vector<
			vector<vector<vector<float>>>> { vector<vector<vector<float>>> {
			vector<vector<float>> { vector<float> { 54, 78, 90, 72, 51 },
					vector<float> { 102, 144, 160, 126, 88 }, vector<float> {
							162, 224, 240, 186, 128 }, vector<float> { 144, 198,
							210, 162, 111 }, vector<float> { 111, 152, 160, 123,
							84 } } } };

//	[[[[  54.   78.   90.   72.   51.]
//	   [ 102.  144.  160.  126.   88.]
//	   [ 162.  224.  240.  186.  128.]
//	   [ 144.  198.  210.  162.  111.]
//	   [ 111.  152.  160.  123.   84.]]]]
	try {
		PlainTensorFactory<float> factory;
		TensorP<float> input = factory.arangeAndInit( { 1, 1, 5, 5 }, 0 );

		TensorP<float> weights = factory.onesAndInit( Shape( { 1, 1, 4, 4 } ) );

		TensorP<float> biases = factory.create( Shape( { 1 } ) );
		biases->init( biasV );

		TensorFactoryP<float> sharedFactory = std::make_shared<
				PlainTensorFactory<float>>( factory );

		Convolution2D<float, float, PlainTensor<float>, PlainTensor<float> > layer(
				"test", LinearActivation<float>::getSharedPointer(), 1, 4, 1,
				PADDING_MODE::SAME, input, sharedFactory, sharedFactory );
		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );
		//activate layer
		layer.feedForward();

		//compare w/ expected
		//create expected tensor
		TensorP<float> expectedOutput = factory.create(
				Shape( { 1, 1, 5, 5 } ) );
		expectedOutput->init( expectedOutputV );
		return finishTest( layer.output(), expectedOutput, __func__ );
	} catch ( const std::bad_alloc& e ) {
		std::cout << "Allocation failed: " << e.what() << std::endl;
		return false;
	}
}

//strides != 1 flips the result image | we get correct output, just flipped.
bool convTest_strides_samePadding_1(){
	cout << "Running " << __func__ << flush;

	vector<vector<vector<vector<long>>>> inputVolume { vector<
			vector<vector<long>>> { vector<vector<long>> { 16, vector<long>( 16,
			1 ) } } };
	vector<vector<vector<vector<long>>>> weightsV {
			vector<vector<vector<long>>> { vector<vector<long>>( 3,
					vector<long> { 1, 1, 1 } ) } };
	vector<long> biasV { 1 };

	vector<vector<vector<vector<long>>>> expectedOutputV { vector<
			vector<vector<long>>> { vector<vector<long>> { { 100, 100, 100, 100,
			100, 100, 100, 49 }, { 100, 100, 100, 100, 100, 100, 100, 49 }, {
			100, 100, 100, 100, 100, 100, 100, 49 }, { 100, 100, 100, 100, 100,
			100, 100, 49 }, { 100, 100, 100, 100, 100, 100, 100, 49 }, { 100,
			100, 100, 100, 100, 100, 100, 49 }, { 100, 100, 100, 100, 100, 100,
			100, 49 }, { 49, 49, 49, 49, 49, 49, 49, 25 }
	} } };
	/*
	 [[[[ 100.  100.  100.  100.  100.  100.  100.   49.]
	 [ 100.  100.  100.  100.  100.  100.  100.   49.]
	 [ 100.  100.  100.  100.  100.  100.  100.   49.]
	 [ 100.  100.  100.  100.  100.  100.  100.   49.]
	 [ 100.  100.  100.  100.  100.  100.  100.   49.]
	 [ 100.  100.  100.  100.  100.  100.  100.   49.]
	 [ 100.  100.  100.  100.  100.  100.  100.   49.]
	 [  49.   49.   49.   49.   49.   49.   49.   25.]]]]
	 * */

	try {

		PlainTensorFactory<long> factory;
		TensorP<long> input = factory.create( Shape( { 1, 1, 16, 16 } ) );
		input->init( inputVolume );
		TensorP<long> weights = factory.create( Shape( { 1, 1, 3, 3 } ) );
		weights->init( weightsV );
		TensorP<long> biases = factory.create( Shape( { 1 } ) );
		biases->init( biasV );

		TensorFactoryP<long> sharedFactory = std::make_shared<
				PlainTensorFactory<long>>( factory );

		Convolution2D<long, long, PlainTensor<long>, PlainTensor<long>> layer(
				"test", SquareActivation<long>::getSharedPointer(),
				1, 3, 2, PADDING_MODE::SAME,
				input, sharedFactory, sharedFactory );
		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );
		//activate layer
		layer.feedForward();

		//compare w/ expected
		//create expected tensor
		TensorP<long> expectedOutput = factory.create( Shape( { 1, 1, 8, 8 } ) );
		expectedOutput->init( expectedOutputV );
		return finishTest( layer.output(), expectedOutput, __func__ );
	} catch ( const std::bad_alloc& e ) {
		cout << "Allocation failed: " << e.what() << flush;
		return false;
	}
}

bool convTest_multiple_channels_non_unit_stride_same_padding_1(){
	cout << __func__ << flush;

	vector<vector<vector<vector<long>>>> inputVolume { vector<
			vector<vector<long>>> { 3, vector<vector<long>> { 16, vector<long>(
			16,
			1 ) } } };
	vector<vector<vector<vector<long>>>> weightsV {
			vector<vector<vector<long>>> { 3, vector<vector<long>>( 3,
					vector<long> { 1, 1, 1 } ) } };
	vector<long> biasV { 1 };

//	[[[[784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [361. 361. 361. 361. 361. 361. 361. 169.]]]]
	vector<vector<vector<vector<long>>>> expectedOutputV { vector<
			vector<vector<long>>> { vector<vector<long>> { { 784, 784, 784, 784,
			784, 784, 784, 361 }, { 784, 784, 784, 784, 784, 784, 784, 361 }, {
			784, 784, 784, 784, 784, 784, 784, 361 }, { 784, 784, 784, 784, 784,
			784, 784, 361 }, { 784, 784, 784, 784,
			784, 784,
			784, 361 }, { 784, 784, 784, 784, 784, 784, 784, 361 }, { 784, 784,
			784, 784, 784, 784, 784, 361 }, { 361, 361, 361, 361, 361, 361, 361,
			169 } } } };

	try {
		PlainTensorFactory<long> factory;

		TensorP<long> input = factory.create( Shape( { 1, 3,
				16, 16 } ) );
		input->init( inputVolume );
		TensorP<long> weights = factory.create( Shape( { 1, 3,
				3, 3 } ) );
		weights->init( weightsV );
		TensorP<long> biases = factory.create( Shape( { 1 } ) );
		biases->init( biasV );

		TensorFactoryP<long> sharedFactory = std::make_shared<
				PlainTensorFactory<long>>( factory );

		//may have wrong input w/ 1,3,1
		Convolution2D<long, long, PlainTensor<long>, PlainTensor<long>> layer( "test",
				SquareActivation<long>::getSharedPointer(), 1, 3, 2,
				PADDING_MODE::SAME,
				input, sharedFactory, sharedFactory );
		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );
		//activate layer
		layer.feedForward();

		//compare w/ expected
		//create expected tensor
		TensorP<long> expectedOutput = factory.create( Shape( { 1, 1, 8, 8 } ) );
		expectedOutput->init( expectedOutputV );
		return finishTest( layer.output(), expectedOutput,
				__func__ );
	} catch ( const std::bad_alloc& e ) {
		cout << "Allocation failed: " << e.what() << flush;
		return false;
	}
}

bool convTest_multiple_filters__multiple_channels_non_unit_stride_same_padding_1(){
	cout << __func__ << flush;

	vector<vector<vector<vector<long>>>> inputVolume { vector<vector<vector<long>>> ( 3, vector<vector<long>> ( 16, vector<long>(16, 1 ) ) ) };
	vector<vector<vector<vector<long>>>> weightsV ( 2, vector<vector<vector<long>>> ( 3, vector<vector<long>>( 3, vector<long> { 1, 1, 1 } ) ) );
	vector<long> biasV { 1,1 };

//	[[[[784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [361. 361. 361. 361. 361. 361. 361. 169.]]
//
//	  [[784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [784. 784. 784. 784. 784. 784. 784. 361.]
//	   [361. 361. 361. 361. 361. 361. 361. 169.]]]]

	vector<vector<vector<vector<long>>>> expectedOutputV { vector<
			vector<vector<long>>> (2, vector<vector<long>> {
				{ 784, 784, 784, 784, 784, 784, 784, 361 },
				{ 784, 784, 784, 784, 784, 784, 784, 361 },
				{ 784, 784, 784, 784, 784, 784, 784, 361 },
				{ 784, 784, 784, 784, 784, 784, 784, 361 },
			    { 784, 784, 784, 784, 784, 784, 784, 361 },
				{ 784, 784, 784, 784, 784, 784, 784, 361 },
				{ 784, 784, 784, 784, 784, 784, 784, 361 },
				{ 361, 361, 361, 361, 361, 361, 361, 169 } } ) };

	try {
		PlainTensorFactory<long> factory;

		TensorP<long> input = factory.create( Shape( { 1, 3, 16, 16 } ) );
		input->init( inputVolume );
		TensorP<long> weights = factory.create( Shape( { 2, 3, 3, 3 } ) );
		weights->init( weightsV );
		TensorP<long> biases = factory.create( Shape( { 2 } ) );
		biases->init( biasV );

		TensorFactoryP<long> sharedFactory = std::make_shared<
				PlainTensorFactory<long>>( factory );

		Convolution2D<long, long, PlainTensor<long>, PlainTensor<long>> layer( "test",
				SquareActivation<long>::getSharedPointer(), 2, 3, 2,
				PADDING_MODE::SAME,
				input, sharedFactory, sharedFactory );
		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );
		//activate layer
		layer.feedForward();

		//compare w/ expected
		//create expected tensor
		TensorP<long> expectedOutput = factory.create( Shape( {
				1, 2, 8, 8 } ) );
		expectedOutput->init( expectedOutputV );
		return finishTest( layer.output(), expectedOutput, __func__ );
	} catch ( const std::bad_alloc& e ) {
		cout << "Allocation failed: " << e.what() << flush;
		return false;
	}
}

//bool convTest_batches_multiple_filters__multiple_channels_non_unit_stride_same_padding_1(){
//
//	cout << "Running convTest_batches_multiple_filters__multiple_channels_non_unit_stride_same_padding_1" << "  " << flush;
//
//	vector<vector<vector<long>>> inputVolume( 4, vector< vector<long>>( 3, vector<long>( 16*16, 1 ) ) );
//	vector<vector<int>> filters(2*3,vector<int>{1,1,1, 1,1,1, 1,1,1, 1});
//	vector<vector<vector<long>>> filteredImages;
//
//	int noFilter = 2;
//
////	[[[[784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [361. 361. 361. 361. 361. 361. 361. 169.]]
////
////	  [[784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [361. 361. 361. 361. 361. 361. 361. 169.]]]
////
////
////	 [[[784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [361. 361. 361. 361. 361. 361. 361. 169.]]
////
////	  [[784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [361. 361. 361. 361. 361. 361. 361. 169.]]]
////
////
////	 [[[784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [361. 361. 361. 361. 361. 361. 361. 169.]]
////
////	  [[784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [361. 361. 361. 361. 361. 361. 361. 169.]]]
////
////
////	 [[[784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [361. 361. 361. 361. 361. 361. 361. 169.]]
////
////	  [[784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [784. 784. 784. 784. 784. 784. 784. 361.]
////	   [361. 361. 361. 361. 361. 361. 361. 169.]]]]
//
//	vector<vector<vector<long>>> expectedOutput( 4, vector<vector<long>>( 2, vector<long>{
//		784, 784, 784, 784, 784, 784, 784, 361,
//		784, 784, 784, 784, 784, 784, 784, 361,
//		784, 784, 784, 784, 784, 784, 784, 361,
//		784, 784, 784, 784, 784, 784, 784, 361,
//		784, 784, 784, 784, 784, 784, 784, 361,
//		784, 784, 784, 784, 784, 784, 784, 361,
//		784, 784, 784, 784, 784, 784, 784, 361,
//		361, 361, 361, 361, 361, 361, 361, 169,
//	} ) );
//
//	convOperationPlainParallel<long, int>(2, 2, 3, 16, 16, SAME_PADDING,inputVolume, noFilter, filters, SquareActivation<long>::getSharedPointer(), ref(filteredImages) );
//
//	return finishTest( filteredImages, expectedOutput, __func__  );
//}
//
//bool convTest1_samePad_floats(){
//
//	cout << "Running " << __func__ << "  " << flush;
//
//	cout << setprecision(std::numeric_limits<long double>::digits10 + 1);
//
//	mnist::MNIST_dataset<uint8_t,uint8_t> dataset = loadMNIST("src/mnist");
//
//	vector<vector<double>> filtersConv1;
//	vector<vector<double>> filtersConv2;
//	vector<vector<double>> weightsVectorFC1;
//	vector<double> biasesFC1;
//	vector<vector<double>> weightsVectorFC2;
//	vector<double> biasesFC2;
//
//	getWeights<double>( filtersConv1, filtersConv2, weightsVectorFC1, biasesFC1, weightsVectorFC2, biasesFC2, "src/keras/data/float_saved_weights/");
//
//	// get the first image
//	vector<vector<vector<double>>> input{ vector<vector<double>>{ vector<double>() } };
//	for( auto i : dataset.test_images[0] )
//		input[0][0].push_back(i);
//
//	// normalize
//	div( input, 255. );
//
//	// compare input with python
//	vector<vector<vector<double>>> reference = loadFromDataset<double>( 1, 28*28, "src/keras/data/float_saved_weights/layer_output/input.txt" );
//	compareConvOutput<double,double>( input, reference );
//	reference.clear();
//
//	// run the first conv layer on one image
//	vector<vector<vector<double>>> convOut;
//	convOperationPlainParallel<double, double>(2, 2, 5, 28, 28, SAME_PADDING, input, 32, filtersConv1, SquareActivation<double>::getSharedPointer(), convOut);
//	reference = loadConvLayerOutput<double>( "conv2d_1.txt", 1, 32, 14*14, false, "src/keras/data/float_saved_weights/layer_output/" );
//	compareConvOutput<double,double>( convOut, reference );
//
//
//	return false;
//
//}

//arange input same padding even sized kernels
bool convRangeSameEvenTest1() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<long> dataFactory, weightFactory;

	//setup input
	TensorP<long> input = dataFactory.arangeAndInit( { 1, 1, 5, 5 } );

	//setup weights & biases
	vector<long> biasV { 0 };
	TensorP<long> biases = dataFactory.create( Shape( { 1 } ) );
	biases->init( biasV );
	TensorP<long> weights = dataFactory.arangeAndInit( { 1, 1, 2, 2 } );

	//setup expected output
	std::string kerasOutput =
			"[[[[  51.   71.   35.][ 151.  171.   75.][  65.   71.   25.]]]]";
	TensorP<long> expectedOutput = createTensorFromKeras<long>( { 1, 1, 3, 3 },
			kerasOutput );

	TensorFactoryP<long> sharedDataFactory = std::make_shared<
			PlainTensorFactory<long>>( dataFactory );
	TensorFactoryP<long> sharedWeightFactory = std::make_shared<
			PlainTensorFactory<long>>( weightFactory );

	//setup layer and run
	int noFilters = 1, filterSize = 2, stride = 2;
	Convolution2D<long, long, PlainTensor<long>, PlainTensor<long>> layer(
			"test", LinearActivation<long>::getSharedPointer(), noFilters,
			filterSize, stride, PADDING_MODE::SAME, input, sharedDataFactory,
			sharedWeightFactory );
	layer.output()->init();
	//set weights/biases of layer
	layer.weights( weights );
	layer.biases( biases );
	//activate layer
	layer.feedForward();

	//compare output w/ expected output
	return finishTest( layer.output(), expectedOutput, __func__ );
}

//arange input same padding odd sized kernels
bool convRangeSameOddTest1() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<long> dataFactory, weightFactory;

	//setup input
	TensorP<long> input = dataFactory.arangeAndInit( { 1, 1, 5, 5 } );

	//setup weights & biases
	vector<long> biasV { 0 };
	TensorP<long> biases = dataFactory.create( Shape( { 1 } ) );
	biases->init( biasV );
	TensorP<long> weights = dataFactory.arangeAndInit( { 1, 1, 3, 3 } );

	//setup expected output
	std::string kerasOutput =
			"[[[[ 128.  241.  184.][ 441.  681.  453.][ 320.  457.  280.]]]]";
	TensorP<long> expectedOutput = createTensorFromKeras<long>( { 1, 1, 3, 3 },
			kerasOutput );

	TensorFactoryP<long> sharedDataFactory = std::make_shared<
			PlainTensorFactory<long>>( dataFactory );
	TensorFactoryP<long> sharedWeightFactory = std::make_shared<
			PlainTensorFactory<long>>( weightFactory );

	//setup layer and run
	int noFilters = 1, filterSize = 3, stride = 2;
	Convolution2D<long, long, PlainTensor<long>, PlainTensor<long>> layer(
			"test", LinearActivation<long>::getSharedPointer(), noFilters,
			filterSize, stride, PADDING_MODE::SAME, input, sharedDataFactory,
			sharedWeightFactory );
	layer.output()->init();
	//set weights/biases of layer
	layer.weights( weights );
	layer.biases( biases );
	//activate layer
	layer.feedForward();

	//compare output w/ expected output
	return finishTest( layer.output(), expectedOutput, __func__ );
}

bool convMultipleFiltersRangeSameOddTest1() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<long> dataFactory, weightFactory;

	//setup input
	TensorP<long> input = dataFactory.arangeAndInit( { 1, 1, 5, 5 } );

	//setup weights & biases
	vector<long> biasV { 0, 0 };
	TensorP<long> biases = dataFactory.create( Shape( { 2 } ) );
	biases->init( biasV );
	std::string weightKeras =
			"[[  1.   3.   5.][  7.   9.  11.][ 13.  15.  17.]][[  2.   4.   6.][  8.  10.  12.][ 14.  16.  18.]]";
	TensorP<long> weights = createTensorFromKeras<long>( { 2, 1, 3, 3 },
			weightKeras );

	//setup expected output
	std::string kerasOutput =
			"[[[[  240.   449.   340.][  813.  1245.   819.][  564.   791.   472.]][[  256.   482.   368.][  882.  1362.   906.][  640.   914.   560.]]]]";
	TensorP<long> expectedOutput = createTensorFromKeras<long>( { 1, 2, 3, 3 },
			kerasOutput );

	TensorFactoryP<long> sharedDataFactory = std::make_shared<
			PlainTensorFactory<long>>( dataFactory );
	TensorFactoryP<long> sharedWeightFactory = std::make_shared<
			PlainTensorFactory<long>>( weightFactory );

	//setup layer and run
	int noFilters = 2, filterSize = 3, stride = 2;
	Convolution2D<long, long, PlainTensor<long>, PlainTensor<long>> layer(
			"test", LinearActivation<long>::getSharedPointer(), noFilters,
			filterSize, stride, PADDING_MODE::SAME, input, sharedDataFactory,
			sharedWeightFactory );
	layer.output()->init();
	//set weights/biases of layer
	layer.weights( weights );
	layer.biases( biases );
	//activate layer
	layer.feedForward();

	//compare output w/ expected output
	return finishTest( layer.output(), expectedOutput, __func__ );
}

bool convMultipleFiltersRangeSameEvenTest1() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<long> dataFactory, weightFactory;

	//setup input
	TensorP<long> input = dataFactory.arangeAndInit( { 1, 1, 5, 5 } );

	//setup weights & biases
	vector<long> biasV { 0, 0 };
	TensorP<long> biases = dataFactory.create( Shape( { 2 } ) );
	biases->init( biasV );
	std::string weightKeras =
			"[[  1.   3.][  5.   7.]][[  2.   4.][  6.  8.]]";
	TensorP<long> weights = createTensorFromKeras<long>( { 2, 1, 2, 2 },
			weightKeras );

	//setup expected output
	std::string kerasOutput =
			"[[[[  86.  118.   55.][ 246.  278.  115.][  87.   95.   25.]][[ 102.  142.   70.][ 302.  342.  150.][ 130.  142.   50.]]]]";
	TensorP<long> expectedOutput = createTensorFromKeras<long>( { 1, 2, 3, 3 },
			kerasOutput );

	TensorFactoryP<long> sharedDataFactory = std::make_shared<
			PlainTensorFactory<long>>( dataFactory );
	TensorFactoryP<long> sharedWeightFactory = std::make_shared<
			PlainTensorFactory<long>>( weightFactory );

	//setup layer and run
	int noFilters = 2, filterSize = 2, stride = 2;
	Convolution2D<long, long, PlainTensor<long>, PlainTensor<long>> layer(
			"test", LinearActivation<long>::getSharedPointer(), noFilters,
			filterSize, stride, PADDING_MODE::SAME, input, sharedDataFactory,
			sharedWeightFactory );
	layer.output()->init();
	//set weights/biases of layer
	layer.weights( weights );
	layer.biases( biases );
	//activate layer
	layer.feedForward();

	//compare output w/ expected output
	return finishTest( layer.output(), expectedOutput, __func__ );
}

bool convMultipleChannelsRangeSameEvenTest1() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<long> dataFactory, weightFactory;

	//setup input
	TensorP<long> input = dataFactory.arangeAndInit( { 1, 2, 5, 5 } );

	//setup weights & biases
	vector<long> biasV { 0 };
	TensorP<long> biases = dataFactory.create( Shape( { 1 } ) );
	biases->init( biasV );
	std::string weightKeras =
			"[[  1.   3.][  5.   7.]][[  2.   4.][  6.  8.]]";
	TensorP<long> weights = createTensorFromKeras<long>( { 1, 2, 2, 2 },
			weightKeras );

	//setup expected output
	std::string kerasOutput =
			"[[[[  688.   760.   325.][ 1048.  1120.   465.][  367.   387.   125.]]]]";
	TensorP<long> expectedOutput = createTensorFromKeras<long>( { 1, 1, 3, 3 },
			kerasOutput );

	TensorFactoryP<long> sharedDataFactory = std::make_shared<
			PlainTensorFactory<long>>( dataFactory );
	TensorFactoryP<long> sharedWeightFactory = std::make_shared<
			PlainTensorFactory<long>>( weightFactory );

	//setup layer and run
	int noFilters = 1, filterSize = 2, stride = 2;
	Convolution2D<long, long, PlainTensor<long>, PlainTensor<long>> layer(
			"test", LinearActivation<long>::getSharedPointer(), noFilters,
			filterSize, stride, PADDING_MODE::SAME, input, sharedDataFactory,
			sharedWeightFactory );
	layer.output()->init();
	//set weights/biases of layer
	layer.weights( weights );
	layer.biases( biases );
	//activate layer
	layer.feedForward();

	//compare output w/ expected output
	return finishTest( layer.output(), expectedOutput, __func__ );
}

bool convMultipleChannelsRangeSameOddTest1() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<long> dataFactory, weightFactory;

	//setup input
	TensorP<long> input = dataFactory.arangeAndInit( { 1, 2, 5, 5 } );

	//setup weights & biases
	vector<long> biasV { 0 };
	TensorP<long> biases = dataFactory.create( Shape( { 1 } ) );
	biases->init( biasV );
	std::string weightKeras =
			"[[  1.   3.   5.][  7.   9.  11.][ 13.  15.  17.]][[  2.   4.   6.][  8.  10.  12.][ 14.  16.  18.]]";
	TensorP<long> weights = createTensorFromKeras<long>( { 1, 2, 3, 3 },
			weightKeras );

	//setup expected output
	std::string kerasOutput =
			"[[[[ 1896.  2881.  1908.][ 3345.  4857.  3075.][ 2004.  2755.  1632.]]]]";
	TensorP<long> expectedOutput = createTensorFromKeras<long>( { 1, 1, 3, 3 },
			kerasOutput );

	TensorFactoryP<long> sharedDataFactory = std::make_shared<
			PlainTensorFactory<long>>( dataFactory );
	TensorFactoryP<long> sharedWeightFactory = std::make_shared<
			PlainTensorFactory<long>>( weightFactory );

	//setup layer and run
	int noFilters = 1, filterSize = 3, stride = 2;
	Convolution2D<long, long, PlainTensor<long>, PlainTensor<long>> layer(
			"test", LinearActivation<long>::getSharedPointer(), noFilters,
			filterSize, stride, PADDING_MODE::SAME, input, sharedDataFactory,
			sharedWeightFactory );
	layer.output()->init();
	//set weights/biases of layer
	layer.weights( weights );
	layer.biases( biases );
	//activate layer
	layer.feedForward();

	//compare output w/ expected output
	return finishTest( layer.output(), expectedOutput, __func__ );
}

bool convMultipleChannelsMultipleFiltersRangeSameEvenTest1() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<long> dataFactory, weightFactory;

	//setup input
	TensorP<long> input = dataFactory.arangeAndInit( { 1, 2, 5, 5 } );

	//setup weights & biases
	vector<long> biasV { 0, 0 };
	TensorP<long> biases = dataFactory.create( Shape( { 2 } ) );
	biases->init( biasV );

	//had to reorganize the weights such that it is first each of the weights' first channel, then all of the weights second channel, etc...
	std::string weightKeras =
			"[[  1.   5.][ 9.  13.]][[  3.   7.][ 11.  15.]][[  2.   6.][ 10.  14.]][[  4.   8.][ 12.  16.]]";
	TensorP<long> weights = createTensorFromKeras<long>( { 2, 2, 2, 2 },
			weightKeras );

	//setup expected output
	std::string kerasOutput =
			"[[[[ 1244.  1372.   570.][ 1884.  2012.   810.][  598.   630.   175.]][[ 1376.  1520.   650.][ 2096.  2240.   930.][  734.   774.   250.]]]]";
	TensorP<long> expectedOutput = createTensorFromKeras<long>( { 1, 2, 3, 3 },
			kerasOutput );

	TensorFactoryP<long> sharedDataFactory = std::make_shared<
			PlainTensorFactory<long>>( dataFactory );
	TensorFactoryP<long> sharedWeightFactory = std::make_shared<
			PlainTensorFactory<long>>( weightFactory );

	//setup layer and run
	int noFilters = 2, filterSize = 2, stride = 2;
	Convolution2D<long, long, PlainTensor<long>, PlainTensor<long>> layer(
			"test", LinearActivation<long>::getSharedPointer(), noFilters,
			filterSize, stride, PADDING_MODE::SAME, input, sharedDataFactory,
			sharedWeightFactory );
	layer.output()->init();
	//set weights/biases of layer
	layer.weights( weights );
	layer.biases( biases );
	//activate layer
	layer.feedForward();

	//compare output w/ expected output
	return finishTest( layer.output(), expectedOutput, __func__ );
}

bool convMultipleChannelsMultipleFiltersRangeSameOddTest1() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<long> dataFactory, weightFactory;

	//setup input
	TensorP<long> input = dataFactory.arangeAndInit( { 1, 2, 5, 5 } );

	//setup weights & biases
	vector<long> biasV { 0, 0 };
	TensorP<long> biases = dataFactory.create( Shape( { 2 } ) );
	biases->init( biasV );

	//had to reorganize the weights such that it is first each of the weights' first channel, then all of the weights second channel, etc...
	std::string weightKeras =
			"[[  1.   5.   9.][ 13.  17.  21.][ 25.  29.  33.]][[  3.   7.  11.][ 15.  19.  23.][ 27.  31.  35.]][[  2.   6.  10.][ 14.  18.  22.][ 26.  30.  34.]][[  4.   8.  12.][ 16.  20.  24.][ 28.  32.  36.]]";
	TensorP<long> weights = createTensorFromKeras<long>( { 2, 2, 3, 3 },
			weightKeras );

	//setup expected output
	std::string kerasOutput =
			"[[[[ 3660.  5546.  3660.][ 6402.  9255.  5826.][ 3756.  5114.  2988.]][[ 3792.  5762.  3816.][ 6690.  9714.  6150.][ 4008.  5510.  3264.]]]]";
	TensorP<long> expectedOutput = createTensorFromKeras<long>( { 1, 2, 3, 3 },
			kerasOutput );

	TensorFactoryP<long> sharedDataFactory = std::make_shared<
			PlainTensorFactory<long>>( dataFactory );
	TensorFactoryP<long> sharedWeightFactory = std::make_shared<
			PlainTensorFactory<long>>( weightFactory );

	//setup layer and run
	int noFilters = 2, filterSize = 3, stride = 2;
	Convolution2D<long, long, PlainTensor<long>, PlainTensor<long>> layer(
			"test", LinearActivation<long>::getSharedPointer(), noFilters,
			filterSize, stride, PADDING_MODE::SAME, input, sharedDataFactory,
			sharedWeightFactory );
	layer.output()->init();
	//set weights/biases of layer
	layer.weights( weights );
	layer.biases( biases );
	//activate layer
	layer.feedForward();

	//compare output w/ expected output
	return finishTest( layer.output(), expectedOutput, __func__ );
}




