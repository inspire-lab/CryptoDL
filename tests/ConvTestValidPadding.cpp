
/*
 * ConvTestValidPadding.cpp
 *
 *  Created on: Jan 23, 2019
 *      Author: robert
 */


#include "CompleteNetworkTests.h"
#include "ConvTest.h"

bool convTest1_validPad(){

	std::cout << __func__ << std::flush;

	std::vector<std::vector<std::vector<std::vector<long>>>> inputVolume { std::vector<std::vector<std::vector<long>>> {
			std::vector<std::vector<long>>( 4, std::vector<long> { 1, 1, 1, 1 } ) } };
	std::vector<std::vector<std::vector<std::vector<long>>>> weightsV { std::vector<std::vector<std::vector<long>>> {
			std::vector<std::vector<long>>( 3, std::vector<long> { 1, 1, 1 } ) } }; //
	std::vector<long> biasV { 1 };
	std::vector<std::vector<std::vector<std::vector<long>>>> expectedOutputV { std::vector<
			std::vector<std::vector<long>>> { std::vector<std::vector<long>> { std::vector<long> { 100, 100 },
			std::vector<long> { 100, 100 } } } };

//  [[[[100. 100. ]
//     [100. 100. ]]]]
	return executeConvTestValid<long>( __func__, inputVolume, weightsV, biasV,
			expectedOutputV );
}

bool convTest2_validPad(){
	std::cout << __func__ << std::flush;

	std::vector<std::vector<std::vector<std::vector<long>>>> inputVolume { std::vector<std::vector<std::vector<long>>> {
			std::vector<std::vector<long>>( 4, std::vector<long> { 1,
			2, 2, 1 } ) } };
	std::vector<std::vector<std::vector<std::vector<long>>>> weightsV { std::vector<std::vector<std::vector<long>>> {
			std::vector<std::vector<long>>( 3, std::vector<long> { 1, 1, 1 } ) } }; //
	std::vector<long> biasV { 1 };
	std::vector<std::vector<std::vector<std::vector<long>>>> expectedOutputV { std::vector<
			std::vector<std::vector<long>>> { std::vector<std::vector<long>> { std::vector<long> { 256, 256 },
			std::vector<long> { 256, 256 } } } };

//  [[[[256. 256. ]
//     [256. 256. ]]]]
	return executeConvTestValid<long>( __func__, inputVolume, weightsV, biasV,
			expectedOutputV );
}


bool convTest3_validPad(){
	std::cout << __func__ << std::flush;

	std::vector<std::vector<std::vector<std::vector<long>>>> inputVolume { std::vector<std::vector<std::vector<long>>> {
			std::vector<std::vector<long>> { { 2, 2, 3, 2 }, { 2,
			1, 2, 3 }, { 1, 3, 2, 3 }, { 2, 2, 1, 2 } } } };
	std::vector<std::vector<std::vector<std::vector<long>>>> weightsV { std::vector<std::vector<std::vector<long>>> {
			std::vector<std::vector<long>>( 3, std::vector<long> { 1, 1, 1 } ) } }; //
	std::vector<long> biasV { 1 };
	std::vector<std::vector<std::vector<std::vector<long>>>> expectedOutputV { std::vector<
			std::vector<std::vector<long>>> { std::vector<std::vector<long>> { std::vector<long> { 361, 484 },
			std::vector<long> { 289, 400 } } } };
//  [[[[361. 484. ]
//     [289. 400. ]]]]
	return executeConvTestValid<long>( __func__, inputVolume, weightsV, biasV,
			expectedOutputV );
}


bool convTest4_validPad(){
	std::cout << __func__ << std::flush;

	std::vector<std::vector<std::vector<std::vector<long>>>> inputVolume { std::vector<std::vector<std::vector<long>>> {
			std::vector<std::vector<long>> { { 1, 3, 3, 2 }, { 2,
			1, 3, 2 }, { 1, 3, 2, 1 }, { 3, 2, 3, 1 } } } };
	std::vector<std::vector<std::vector<std::vector<long>>>> weightsV { std::vector<std::vector<std::vector<long>>> {
			std::vector<std::vector<long>>( 3, std::vector<long> { 1, 1, 1 } ) } }; //
	std::vector<long> biasV { 1 };
	std::vector<std::vector<std::vector<std::vector<long>>>> expectedOutputV { std::vector<
			std::vector<std::vector<long>>> { std::vector<std::vector<long>> { std::vector<long> { 400, 441 },
			std::vector<long> { 441, 361 } } } };
//  [[[[400. 441. ]
//     [441. 361. ]]]]
	return executeConvTestValid<long>( __func__, inputVolume, weightsV, biasV,
			expectedOutputV );
}


bool convTest5_validPad(){
	std::cout << __func__ << std::flush;

	std::vector<std::vector<std::vector<std::vector<long>>>> inputVolume { std::vector<std::vector<std::vector<long>>> {
			std::vector<std::vector<long>> { { 3, 2, 3, 2 }, { 2,
			4, 1, 3 }, { 2, 2, 1, 3 }, { 1, 3, 2, 3 } } } };
	std::vector<std::vector<std::vector<std::vector<long>>>> weightsV { std::vector<std::vector<std::vector<long>>> {
			std::vector<std::vector<long>>( 3, std::vector<long> { 1, 1, 1 } ) } }; //
	std::vector<long> biasV { 1 };
	std::vector<std::vector<std::vector<std::vector<long>>>> expectedOutputV { std::vector<
			std::vector<std::vector<long>>> { std::vector<std::vector<long>> { std::vector<long> { 441, 484 },
			std::vector<long> { 361, 529 } } } };
//	[[[[441. 484.]
//	   [361. 529.]]]]
	return executeConvTestValid<long>( __func__, inputVolume, weightsV, biasV,
			expectedOutputV );
}

bool convTest_strides_validPad_1(){

	std::cout << __func__ << std::flush;

	std::vector<std::vector<std::vector<std::vector<long>>>> inputVolume { std::vector<std::vector<std::vector<long>>> {
			std::vector<std::vector<long>> { 16, std::vector<long>( 16,
			1 ) } } };
	std::vector<std::vector<std::vector<std::vector<long>>>> weightsV { std::vector<std::vector<std::vector<long>>> {
			std::vector<std::vector<long>>( 3, std::vector<long> { 1, 1, 1 } ) } };
	std::vector<long> biasV { 1 };
	std::vector<std::vector<std::vector<std::vector<long>>>> expectedOutputV { std::vector<
			std::vector<std::vector<long>>> { std::vector<std::vector<long>> { 7, std::vector<long>( 7,
			100 ) } } };

//	[[[[100. 100. 100. 100. 100. 100. 100.]
//	   [100. 100. 100. 100. 100. 100. 100.]
//	   [100. 100. 100. 100. 100. 100. 100.]
//	   [100. 100. 100. 100. 100. 100. 100.]
//	   [100. 100. 100. 100. 100. 100. 100.]
//	   [100. 100. 100. 100. 100. 100. 100.]
//	   [100. 100. 100. 100. 100. 100. 100.]]]]

	try {
		PlainTensorFactory<long> factory;
		TensorP<long> input = factory.create( Shape( { 1, 1, 16, 16 } ) );
		input->init( inputVolume );
		TensorP<long> weights = factory.create( Shape( { 1, 1, 3, 3 } ) );
		weights->init( weightsV );
		TensorP<long> biases = factory.create( Shape( { 1 } ) );
		biases->init( biasV );

		//may have wrong input w/ 1,3,1
		Convolution2D<long, long, PlainTensor<long>, PlainTensor<long>> layer( "test",
				SquareActivation<long>::getSharedPointer(), 1, 3, 2,
				PADDING_MODE::VALID, input, &factory, &factory );
		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );

		//testing to see if output was created


		//activate layer
		layer.feedForward();

		//compare w/ expected
		//create expected tensor
		TensorP<long> expectedOutput = factory.create( Shape( { 1, 1, 7, 7 } ) );
		expectedOutput->init( expectedOutputV );
		return finishTest( layer.output(), expectedOutput, __func__ );
	} catch ( const std::bad_alloc& e ) {
		std::cout << "Allocation failed: " << e.what() << std::flush;
		return false;
	}
}

bool convTest_multiple_channels_non_unit_stride_validPad_1(){
	std::cout << __func__ << std::flush;

	std::vector<std::vector<std::vector<std::vector<long>>>> inputVolume { std::vector<std::vector<std::vector<long>>>(
			3, std::vector<std::vector<long>> { 16, std::vector<long>( 16, 1 ) } ) };
	std::vector<std::vector<std::vector<std::vector<long>>>> weightsV { std::vector<std::vector<std::vector<long>>>( 3,
			std::vector<std::vector<long>>( 3, std::vector<long> { 1, 1, 1 } ) ) };
	std::vector<long> biasV { 1 };
	std::vector<std::vector<std::vector<std::vector<long>>>> expectedOutputV { std::vector<
			std::vector<std::vector<long>>> { std::vector<std::vector<long>> { 7, std::vector<long>( 7, 784 ) } } };

//	[[[[784. 784. 784. 784. 784. 784. 784.]
//	   [784. 784. 784. 784. 784. 784. 784.]
//	   [784. 784. 784. 784. 784. 784. 784.]
//	   [784. 784. 784. 784. 784. 784. 784.]
//	   [784. 784. 784. 784. 784. 784. 784.]
//	   [784. 784. 784. 784. 784. 784. 784.]
//	   [784. 784. 784. 784. 784. 784. 784.]]]]
	try {
		PlainTensorFactory<long> factory;
		TensorP<long> input = factory.create( Shape( { 1, 3, 16, 16 } ) );
		input->init( inputVolume );
		TensorP<long> weights = factory.create( Shape( { 1, 3, 3, 3 } ) );
		weights->init( weightsV );
		TensorP<long> biases = factory.create( Shape( { 1 } ) );
		biases->init( biasV );

		//may have wrong input w/ 1,3,1
		Convolution2D<long, long, PlainTensor<long>, PlainTensor<long>> layer( "test",
				SquareActivation<long>::getSharedPointer(), 1, 3, 2,
				PADDING_MODE::VALID, input, &factory, &factory );
		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );
		//activate layer
		layer.feedForward();

		//compare w/ expected
		//create expected tensor
		TensorP<long> expectedOutput = factory.create( Shape( { 1, 1, 7, 7 } ) );
		expectedOutput->init( expectedOutputV );
		return finishTest( layer.output(), expectedOutput, __func__ );
	} catch ( const std::bad_alloc& e ) {
		std::cout << "Allocation failed: " << e.what() << std::flush;
		return false;
	}
//	convOperationPlainParallel<long, int>(2, 2, 3, 16, 16, VALID_PADDING,inputVolume, noFilter, filters, SquareActivation<long>::getSharedPointer(), ref(filteredImages) );
//	return finishTest( filteredImages, expectedOutput, __func__  );
}

//bool convTest_multiple_filters__multiple_channels_non_unit_stride_validPad_1(){
//
//	std::cout << "Running convTest_multiple_filters__multiple_channels_non_unit_stride_validPad_1" << "  " << std::flush;
//
//	std::vector<std::vector<std::vector<long>>> inputVolume{ std::vector< std::vector<long>>( 3, std::vector<long>( 16*16, 1 ) ) };
//	std::vector<std::vector<int>> filters(2*3,std::vector<int>{1,1,1, 1,1,1, 1,1,1, 1});
//	std::vector<std::vector<std::vector<long>>> filteredImages;
//
//	int noFilter = 2;
//
////	[[[[784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]]
////
////	  [[784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]]]]
//
//	std::vector<std::vector<std::vector<long>>> expectedOutput{ std::vector<std::vector<long>>( 2, std::vector<long>{
//		784, 784, 784, 784, 784, 784, 784,
//		784, 784, 784, 784, 784, 784, 784,
//		784, 784, 784, 784, 784, 784, 784,
//		784, 784, 784, 784, 784, 784, 784,
//		784, 784, 784, 784, 784, 784, 784,
//		784, 784, 784, 784, 784, 784, 784,
//		784, 784, 784, 784, 784, 784, 784
//	} ) };
//
//	convOperationPlainParallel<long, int>(2, 2, 3, 16, 16, VALID_PADDING,inputVolume, noFilter, filters, SquareActivation<long>::getSharedPointer(), ref(filteredImages) );
//	return finishTest( filteredImages, expectedOutput, __func__  );
//}
//
//bool convTest_batches_multiple_filters__multiple_channels_non_unit_stride_validPad_1(){
//
//	std::cout << "Running convTest_batches_multiple_filters__multiple_channels_non_unit_stride_validPad_1" << "  " << std::flush;
//
//	std::vector<std::vector<std::vector<long>>> inputVolume( 4, std::vector< std::vector<long>>( 3, std::vector<long>( 16*16, 1 ) ) );
//	std::vector<std::vector<int>> filters(2*3,std::vector<int>{1,1,1, 1,1,1, 1,1,1, 1});
//	std::vector<std::vector<std::vector<long>>> filteredImages;
//
//	int noFilter = 2;
//
////	[[[[784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]]
////
////	  [[784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]]]
////
////
////	 [[[784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]]
////
////	  [[784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]]]
////
////
////	 [[[784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]]
////
////	  [[784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]]]
////
////
////	 [[[784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]]
////
////	  [[784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]
////	   [784. 784. 784. 784. 784. 784. 784.]]]]
//
//
//	std::vector<std::vector<std::vector<long>>> expectedOutput( 4, std::vector<std::vector<long>>( 2, std::vector<long>{
//		784, 784, 784, 784, 784, 784, 784,
//		784, 784, 784, 784, 784, 784, 784,
//		784, 784, 784, 784, 784, 784, 784,
//		784, 784, 784, 784, 784, 784, 784,
//		784, 784, 784, 784, 784, 784, 784,
//		784, 784, 784, 784, 784, 784, 784,
//		784, 784, 784, 784, 784, 784, 784,
//	} ) );
//
//	convOperationPlainParallel<long, int>(2, 2, 3, 16, 16, VALID_PADDING,inputVolume, noFilter, filters, SquareActivation<long>::getSharedPointer(), ref(filteredImages) );
//	return finishTest( filteredImages, expectedOutput, __func__  );
//
//}
//
////mpz_class
//typedef float t_type;
//bool convTest_Layer2(){
//	std::cout << "Running " << __func__ << std::flush;
//	std::vector<std::vector<std::vector<t_type>>> filteredImages;//our output
//
//	//load layer 1 and 2 outputs, layer 1 out will be our input and layer 2 will be our expected results
//	std::vector<std::vector<std::vector<t_type>>> inputVolume = loadConvLayerOutput<t_type>("conv2d_1.txt", 1, 32, 14*14, false, "src/keras/data/all_ones/");
//	std::vector<std::vector<std::vector<t_type>>> expectedOutput = loadConvLayerOutput<t_type>("conv2d_2.txt", 1, 64, 5*5, false, "src/keras/data/all_ones/");
//
//	//generate filters of all-ones
//	std::vector<std::vector<t_type>> filters(64*32, std::vector<t_type>(26, 1));
//
//	int noFilter = 64;
//	//int stridex, int stridey, int filterSize, int width, int height, int paddingMode
//	convOperationPlainParallel<t_type, t_type>(2, 2, 5, 14, 14, VALID_PADDING, inputVolume, noFilter, filters, SquareActivation<t_type>::getSharedPointer(), ref(filteredImages) );
//
//	//run conv op
//	compareConvOutput(filteredImages, expectedOutput, false);
//
//	std::cout << "Avg on mag for input is: \t" << computeAvgMag<t_type>(inputVolume) << std::flush<< std::flush;
//	std::cout << "Avg on mag for expected is: \t" << computeAvgMag<t_type>(expectedOutput) << std::flush;
//	std::cout << "Avg on mag for actual is: \t" << computeAvgMag<t_type>(filteredImages) << std::flush;
//	std::cout << "out ref: \t" << expectedOutput[0][0] << std::flush;
//	std::cout << "out out: \t" << filteredImages[0][0] << std::flush;
//
//	return false;
//	//return finishTest( filteredImages, expectedOutput, __func__  );
//}


bool convTest1_validPad_evenfilterszie() {

	std::cout << __func__ << std::flush;

	std::vector<std::vector<std::vector<std::vector<long>>>> inputVolume { std::vector<std::vector<std::vector<long>>> {
			std::vector<std::vector<long>> { std::vector<long> { 0, 1, 2, 3 }, std::vector<long> { 4, 5, 6, 7 },
					std::vector<long> { 8, 9, 10, 11 }, std::vector<long> { 12, 13, 14, 15 } } } };
	std::vector<std::vector<std::vector<std::vector<long>>>> weightsV { std::vector<std::vector<std::vector<long>>> {
			std::vector<std::vector<long>>( 2, std::vector<long> { 1, 1 } ) } }; //
	std::vector<long> biasV { 1 };
	std::vector<std::vector<std::vector<std::vector<long>>>> expectedOutputV { std::vector<
			std::vector<std::vector<long>>> { std::vector<std::vector<long>> { std::vector<long> { 11, 15, 19 },
			std::vector<long> { 27, 31, 35 }, std::vector<long> { 43, 47, 51 } } } };

//	[[[[11. 15. 19.]
//	   [27. 31. 35.]
//	   [43. 47. 51.]]]]

	PlainTensorFactory<long> factory;
	TensorP<long> input = factory.create( Shape( { 1, 1, 4, 4 } ) );

	input->init( inputVolume );

	TensorP<long> weights = factory.create( Shape( { 1, 1, 2, 2 } ) );
	weights->init( weightsV );
	TensorP<long> biases = factory.create( Shape( { 1 } ) );
	biases->init( biasV );

	Convolution2D<long, long, PlainTensor<long>, PlainTensor<long>> layer( "test",
			LinearActivation<long>::getSharedPointer(), 1, 2, 1,
			PADDING_MODE::VALID, input, &factory, &factory );
	layer.output()->init();
	//set weights/biases of layer
	layer.weights( weights );
	layer.biases( biases );
	//activate layer
	layer.feedForward();

	//compare w/ expected
	//create expected tensor
	TensorP<long> expectedOutput = factory.create( Shape( { 1, 1, 3, 3 } ) );
	expectedOutput->init( expectedOutputV );
	return finishTest( layer.output(), expectedOutput, __func__ );

}

bool convTest_validPad_secondLayer_cryptonet() {
	std::cout << "Running " << __func__ << " " << std::flush;
	/// load the data set
	mnist::MNIST_dataset<uint8_t, uint8_t> dataset = loadMNIST( "src/mnist" );
	int batchSize = 1;

	PlainTensorFactory<float> ptFactory;

	/// crop the testdata to fit our batchsize
	std::vector<std::vector<uint8_t>> X( dataset.test_images.begin(), dataset.test_images.begin() + batchSize );
	std::vector<uint8_t> Y( dataset.test_labels.begin(), dataset.test_labels.begin() + batchSize );

	std::vector<TensorP<float>> referenceOutPuts;

	auto conv1OutV = loadConvLayerOutput1<float>( "conv2d_1.txt", batchSize, 5, 14 * 14, false,
			"src/experiments/cryptonet/reference_output/" );
	TensorP<float> conv1Out = ptFactory.create( Shape( { batchSize, 5, 14 * 14 } ) );
	conv1Out->init( conv1OutV );
	conv1Out->reshape( Shape( { batchSize, 5, 14, 14 } ) );

	auto conv2OutV = loadConvLayerOutput1<float>( "conv2d_2.txt", batchSize, 10, 25, false,
			"src/experiments/cryptonet/reference_output/" );


	//	uint batchSize = ctxtFactory.batchsize();

	std::cout << "PLAINTEXTEXTTT" << std::endl;

	TensorP<float> inputplain = ptFactory.create( Shape( { batchSize, 1, 28, 28 } ) );

	Model<float, float, PlainTensor<float>, PlainTensor<float> > plainNet(
			MemoryUsage::greedy,
			&ptFactory, &ptFactory );

	plainNet.addLayer(
			std::make_shared<Convolution2D<float, float, PlainTensor<float>, PlainTensor<float> >>(
					"conv2d_1", SquareActivation<float>::getSharedPointer(), 5, 5, 2, PADDING_MODE::SAME,
					inputplain, &ptFactory, &ptFactory ) );

	//conv2
	plainNet.addLayer(
			std::make_shared<Convolution2D<float, float, PlainTensor<float>, PlainTensor<float> >>(
					"conv2d_2", SquareActivation<float>::getSharedPointer(), 1, 5, 2, PADDING_MODE::VALID, &ptFactory, &ptFactory ) );

	plainNet.mLayers[ 0 ]->loadWeights( "src/experiments/cryptonet/savedWeights/" );

	auto loadedWeights = loadFilterWeights<float>( "src/experiments/cryptonet/savedWeights/" + plainNet.mLayers[ 1 ]->name(), 10, 5, 5 );

	std::cout << loadedWeights.first.size() << " " << loadedWeights.first[ 0 ].size() << std::endl;

	int filter = 1; // <---------------------------------------------------------------------------------------------------------set filter here
	std::cout << "Running for fitler: " << filter << std::endl;

	std::vector<std::vector<float>> weights( loadedWeights.first.begin() + 5 * filter, loadedWeights.first.begin() + 5 * ( filter + 1 ) );
	std::vector<float> bias { loadedWeights.second[ filter ] };



	TensorP<float> weightTensor = ptFactory.create( Shape( { weights.size(), weights[ 0 ].size() } ) );
	TensorP<float> biasTensor = ptFactory.create( Shape( { bias.size() } ) );

	weightTensor->init( weights );
	biasTensor->init( bias );



	auto secondLayer = plainNet.mLayers[ 1 ];

	weightTensor->reshape( Shape( { 1, 5, 5, 5 } ) );
	secondLayer->weights( weightTensor );
	secondLayer->biases( biasTensor );

	std::cout << *secondLayer->weights()<< std::endl;

	plainNet.feedInputTensor( X, Y );
	for ( uint i = 0; i < plainNet.mLayers.size(); ++i ) {
		plainNet.mLayers[ i ]->feedForward();
//		TensorP<double> refOut = referenceOutPuts[ i ];
//		finishTest( plainNet.mLayers[ i ]->output(), refOut, plainNet.mLayers[ i ]->name() );
	}

	std::cout << "Output of the second filter of the second layer. First layer is our impl." << std::endl;
	std::cout << *secondLayer->output() << std::endl;


	secondLayer->input( conv1Out );
	secondLayer->output()->init();
//	secondLayer->feedForward();
	std::cout << "Output of the second filter of the second layer. First layer is from keras" << std::endl;
	std::cout << *secondLayer->output() << std::endl;


	std::vector<float> refernecoutFilter2 { 0.005582081153988838, 0.00022766864276491106, 0.08217478543519974, 0.06357406079769135, 0.0029854874592274427, 0.05164581909775734,
			0.09968076646327972, 0.1559268981218338, 0.13820761442184448, 0.035070616751909256, 0.09574401378631592, 0.36529847979545593, 1.7814393043518066, 0.17894013226032257,
			0.10129860043525696, 0.004178666044026613, 0.03821807727217674, 0.07179886102676392, 2.4447303076158278e-05, 0.007837743498384953, 0.016971617937088013,
			0.33495232462882996, 0.06190767511725426, 0.05243825912475586, 0.0016033663414418697 };

	std::vector<float> ref = conv2OutV[0][filter];

	auto refFilterTensor = ptFactory.create( Shape( { 25 } ) );
	refFilterTensor->init( ref );
	refFilterTensor->reshape( Shape( { 1, 1, 5, 5 } ) );
	std::cout << *refFilterTensor << std::endl;

	std::cout << "one filter and one filter  only " << std::endl;
	std::cout << "one filter and one filter  only " << std::endl;
	std::cout << "one filter and one filter  only " << std::endl;



	std::vector<float> anotherReferenceOutput{7.3465868e-03,1.5431236e-03,4.5236307e-03,9.9081406e-03,1.5470762e-03,
		6.0282354e-03,5.1692654e-03,5.6514791e-03,1.2532925e-02,2.1287359e-03,
		3.9021659e-03,4.8574642e-03,7.3417276e-03,1.1314017e-03,8.9970708e-04,
		4.1548111e-03,1.2865170e-02,2.5430718e-03,5.4269453e-04,2.6906491e-03,
		5.4458566e-03,2.0852113e-02,2.2296304e-02,1.1482146e-05,9.8362856e-04};

	std::cout<<  conv1OutV[0][0] << std::endl;

	TensorP<float> anotherInput = ptFactory.create( Shape( { 14 * 14 } ) );
	anotherInput->init(conv1OutV[0][0]);
	anotherInput->reshape(Shape( { 1,1,14,14 } ));

	Convolution2D<float, float, PlainTensor<float>, PlainTensor<float>> testLayer(	"conv2d_2", SquareActivation<float>::getSharedPointer(), 1, 5, 2, PADDING_MODE::VALID, anotherInput, &ptFactory, &ptFactory );
	testLayer.output()->init();

	//init weights
	weights  = std::vector<std::vector<float>>( loadedWeights.first.begin() + 5, loadedWeights.first.begin() + 6 );

	auto testLayerWeights = ptFactory.create( Shape( { 1, 25 } ) );
	testLayerWeights->init(weights);
	testLayerWeights->reshape(Shape( {1,1,5,5}  ) );
	testLayer.weights( testLayerWeights );


	std::vector<float> anotherbias { loadedWeights.second[ 1 ] };
	TensorP<float> anotherbiasTensor = ptFactory.create( Shape( { bias.size() } ) );
	anotherbiasTensor->init( anotherbias );
	testLayer.biases( anotherbiasTensor );

	testLayer.feedForward();
//	TensorP<float> anotherReferenceOutputTensor= ptFactory.create( Shape( { 25 } ) );

	std::cout << *testLayer.output() << std::endl;
	std::cout << anotherReferenceOutput << std::endl;



	std::cout << "and another one" << std::endl<< std::endl<< std::endl<< std::endl<< std::endl;

	std::vector<float> inputVector3{   5.22011367e-04, 5.22011367e-04, 5.22011367e-04, 5.22011367e-04, 5.22011367e-04,
		   5.22011367e-04, 5.22011367e-04, 5.22011367e-04, 5.22011367e-04, 5.22011367e-04,
		   5.22011367e-04, 5.22011367e-04, 1.51623180e-02, 5.27309999e-02, 1.02164177e-03,
		   5.22011367e-04, 5.22011367e-04, 1.24362083e-02, 2.89851632e-02, 8.37224349e-03,
		   5.22011367e-04, 5.22011367e-04, 2.20898036e-02, 5.45112081e-02, 1.35964030e-04};

	TensorP<float> input3 = ptFactory.create( Shape( { 5 * 5 } ) );
	input3->init(inputVector3);
	input3->reshape(Shape( { 1,1,5,5 } ));

	Convolution2D<float, float, PlainTensor<float>, PlainTensor<float>> testLayer2(	"conv2d_2", LinearActivation<float>::getSharedPointer(), 1, 5, 2, PADDING_MODE::VALID, input3, &ptFactory, &ptFactory );
	testLayer2.output()->init();

	//init weights
	weights  = std::vector<std::vector<float>>( loadedWeights.first.begin() + 5, loadedWeights.first.begin() + 6 );

	auto testLayerWeights2 = ptFactory.create( Shape( { 1, 25 } ) );
	testLayerWeights2->init(weights);
	testLayerWeights2->reshape(Shape( {1,1,5,5}  ) );
	testLayer2.weights( testLayerWeights2 );


	std::vector<float> anotherbias2 { loadedWeights.second[ 1 ] };
	TensorP<float> anotherbiasTensor2 = ptFactory.create( Shape( { bias.size() } ) );
	anotherbiasTensor2->init( anotherbias2 );
	testLayer2.biases( anotherbiasTensor2 );

	testLayer2.feedForward();
//	TensorP<float> anotherReferenceOutputTensor= ptFactory.create( Shape( { 25 } ) );

	std::cout << *testLayer2.output() << std::endl;
	std::cout << -0.08571223 << std::endl;

	std::cout << "weights"<< std::endl;
	std::cout << *testLayerWeights2 << std::endl;
	std::cout << "bias"<< std::endl;
	std::cout << *anotherbiasTensor2 << std::endl;
	std::cout << "innput"<< std::endl;
	std::cout << *input3 << std::endl;

	std::cout << loadedWeights.second << std::endl;





	return false; //finishTest( secondLayer->output(), refFilterTensor, __func__ );
}








































