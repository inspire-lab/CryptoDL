
/*
 * HEBackendTests.cpp
 *
 *  Created on: Mar 3, 2019
 *      Author: robert
 */

#include <vector>
#include "../src/architecture/Tensor.h"
#include "../src/architecture/PlainTensor.h"
#include "../src/architecture/HEBackend/HETensor.h"
#include "../src/architecture/HEBackend/helib/HELIbCipherText.h"
#include "../src/architecture/Layer.h"
#include "../src/data/DatasetOperations.h"
#include "HEBackendTests.h"
#include <helib/NumbTh.h>
#include "CompleteNetworkTests.h"
#include "TestCommons.h"


using namespace std;


bool HE_convTest1_samePad() {

	cout << "Running " << __func__ << " " << endl;

	HELibCipherTextFactory ctxtFactory;

	vector<vector<vector<vector<HELibCipherText>> >> inputVolume { vector<vector<vector<HELibCipherText>>> {
			vector<vector<HELibCipherText>>( 4,
					vector<HELibCipherText> { ctxtFactory.createCipherText( 1 ), ctxtFactory.createCipherText( 1 ),
							ctxtFactory.createCipherText( 1 ), ctxtFactory.createCipherText( 1 ) } ) } };

	vector<vector<vector<vector<long>> >> weightsV { vector<vector<vector<long>>> { vector<vector<long>>( 3, vector<long> { 1, 1, 1 } ) } }; //
	vector<long> biasV { 1 };
	vector<vector<vector<vector<long>> >> expectedOutputV { vector<vector<vector<long>>> {
			vector<vector<long>> {
					vector<long> { 25, 49, 49, 25 },
					vector<long> { 49, 100, 100, 49 },
					vector<long> { 49, 100, 100, 49 },
					vector<long> { 25, 49, 49, 25 } } } };



	try {
		HETensorFactory<HELibCipherText> hetfactory( &ctxtFactory );
		PlainTensorFactory<long> ptFactory;

		TensorP<HELibCipherText> input = hetfactory.create( Shape( { 1, 1, 4, 4 } ) );
		input->init( inputVolume );

		TensorP<long> weights = ptFactory.create( Shape( { 1, 1, 3, 3 } ) );

		weights->init( weightsV );
		TensorP<long> biases = ptFactory.create( Shape( { 1 } ) );
		biases->init( biasV );


		Convolution2D<HELibCipherText, long, HETensor<HELibCipherText>, PlainTensor<long> > layer( "test",
				SquareActivation<HELibCipherText>::getSharedPointer(), 1, 3, 1, PADDING_MODE::SAME, input, &hetfactory,
				&ptFactory );


		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );
		//activate layer
		layer.feedForward();

		//compare w/ expected
		if ( ( (HELibCipherTextFactory*) hetfactory.ciphertextFactory() )->useBFV ) {
			//create expected tensor
			TensorP<long> expectedOutput = ptFactory.create( Shape( { 1, 1, 4, 4 } ) );
			expectedOutput->init( expectedOutputV );
			TensorP<long> decrypted = ( (HETensor<HELibCipherText>*) layer.output().get() )->decryptLong();
			return finishTest<long, long>( decrypted, expectedOutput, __func__ );
		} else {
			//create expected tensor
			TensorP<long> expectedOutput = ptFactory.create( Shape( { 1, 1, 4, 4 } ) );
			expectedOutput->init( expectedOutputV );
			TensorP<double> decrypted = ( (HETensor<HELibCipherText>*) layer.output().get() )->decryptDouble();
			return finishTest<double, long>( decrypted, expectedOutput, __func__ );
		}

	} catch ( const std::bad_alloc& e ) {
		cout << "Allocation failed: " << e.what() << endl;
		return false;
	}

}

bool HE_convTest2_samePad() {

	cout << "Running " << __func__ << " " << endl;
	return true;
	HELibCipherTextFactory ctxtFactory;

	vector<vector<vector<vector<HELibCipherText>> >> inputVolume { vector<
			vector<vector<HELibCipherText>>> { vector<vector<HELibCipherText>>(
			4,
			vector<HELibCipherText> { ctxtFactory.createCipherText( 1 ),
					ctxtFactory.createCipherText( 2 ), ctxtFactory
							.createCipherText( 2 ),
					ctxtFactory.createCipherText( 1 ) } ) } };

	vector<vector<vector<vector<long>>>> weightsV {
			vector<vector<vector<long>>> { vector<vector<long>>( 3,
					vector<long> { 1, 1, 1 } ) } };
	vector<long> biasV { 1 };
	vector<vector<vector<vector<long>> >> expectedOutputV { vector<
			vector<vector<long>>> { vector<vector<long>> { vector<long> { 49,
			121, 121, 49 }, vector<long> { 100, 256, 256, 100 }, vector<long> {
			100, 256, 256, 100 }, vector<long> { 49, 121, 121, 49 } } } };
	try {
		HETensorFactory<HELibCipherText> hetfactory( &ctxtFactory );
		PlainTensorFactory<long> ptFactory;

		TensorP<HELibCipherText> input = hetfactory.create( Shape( { 1, 1, 4,
				4 } ) );
		input->init( inputVolume );

		TensorP<long> weights = ptFactory.create( Shape( { 1, 1, 3, 3 } ) );

		weights->init( weightsV );
		TensorP<long> biases = ptFactory.create( Shape( { 1 } ) );
		biases->init( biasV );

		Convolution2D<HELibCipherText, long, HETensor<HELibCipherText>,
				PlainTensor<long> > layer( "test",
				SquareActivation<HELibCipherText>::getSharedPointer(), 1, 3, 1,
				PADDING_MODE::SAME, input, &hetfactory, &ptFactory );


		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );
		//activate layer
		layer.feedForward();

		//compare w/ expected
		if ( ( (HELibCipherTextFactory*) hetfactory.ciphertextFactory() )
				->useBFV ) {
			//create expected tensor
			TensorP<long> expectedOutput = ptFactory.create(
					Shape( { 1, 1, 4, 4 } ) );
			expectedOutput->init( expectedOutputV );
			TensorP<long> decrypted = ( (HETensor<HELibCipherText>*) layer
					.output().get() )->decryptLong();
			return finishTest<long, long>( decrypted, expectedOutput, __func__ );
		} else {
			//create expected tensor
			TensorP<long> expectedOutput = ptFactory.create(
					Shape( { 1, 1, 4, 4 } ) );
			expectedOutput->init( expectedOutputV );
			TensorP<double> decrypted = ( (HETensor<HELibCipherText>*) layer
					.output().get() )->decryptDouble();
			return finishTest<double, long>( decrypted, expectedOutput,
					__func__ );
		}

	} catch ( const std::bad_alloc& e ) {
		cout << "Allocation failed: " << e.what() << endl;
		return false;
	}

}

bool HE_convTest1_samePad_CKKS() {

	cout << "Running " << __func__ << " " << endl;

	HELibCipherTextFactory ctxtFactory( /*L*/4096, /*m*/1 * 4, /*r*/16 );

	vector<vector<vector<vector<HELibCipherText>> >> inputVolume { vector<vector<vector<HELibCipherText>>> {
			vector<vector<HELibCipherText>>( 4,
					vector<HELibCipherText> { ctxtFactory.createCipherText( 1 ), ctxtFactory.createCipherText( 1 ),
							ctxtFactory.createCipherText( 1 ), ctxtFactory.createCipherText( 1 ) } ) } };

	vector<vector<vector<vector<float>> >> weightsV { vector<vector<vector<float>>> { vector<vector<float>>( 3, vector<float> { .1, .1, .1 } ) } }; //
	vector<float> biasV { 0 };
	vector<vector<vector<vector<float>> >> expectedOutputV { vector<vector<vector<float>>> {
			vector<vector<float>> {
					vector<float> { 0.16000001, 0.36, 0.36, 0.16000001 },
					vector<float> { 0.36, 0.81000006, 0.81000006, 0.36 },
					vector<float> { 0.36, 0.81000006, 0.81000006, 0.36 },
					vector<float> { 0.16000001, 0.36, 0.36, 0.16000001 } } } };

	try {
		HETensorFactory<HELibCipherText> hetfactory( &ctxtFactory );
		PlainTensorFactory<float> ptFactory;

		TensorP<HELibCipherText> input = hetfactory.create( Shape( { 1, 1, 4, 4 } ) );
		input->init( inputVolume );

		TensorP<float> weights = ptFactory.create( Shape( { 1, 1, 3, 3 } ) );

		weights->init( weightsV );
		TensorP<float> biases = ptFactory.create( Shape( { 1 } ) );
		biases->init( biasV );

		Convolution2D<HELibCipherText, float, HETensor<HELibCipherText>, PlainTensor<float> > layer( "test",
				SquareActivation<HELibCipherText>::getSharedPointer(), 1, 3, 1, PADDING_MODE::SAME, input, &hetfactory,
				&ptFactory );

		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );
		//activate layer
		layer.feedForward();

		//create expected tensor
		TensorP<float> expectedOutput = ptFactory.create( Shape( { 1, 1, 4, 4 } ) );
		expectedOutput->init( expectedOutputV );
		TensorP<double> decrypted = ( (HETensor<HELibCipherText>*) layer.output().get() )->decryptDouble();
		return finishTest<double, float>( decrypted, expectedOutput, __func__ );

	} catch ( const std::bad_alloc& e ) {
		cout << "Allocation failed: " << e.what() << endl;
		return false;
	}

}

bool HE_convTest1_samePadBatch_CKKS() {

	cout << "Running " << __func__ << " " << endl;

	HELibCipherTextFactory ctxtFactory( /*L*/4096, /*m*/2 * 4, /*r*/32 );

	vector<vector<vector<vector<HELibCipherText>> >> inputVolume { vector<vector<vector<HELibCipherText>>> {
			vector<vector<HELibCipherText>>( 4,
					vector<HELibCipherText> { ctxtFactory.createCipherText( vector<long> { 1, 1 } ), ctxtFactory.createCipherText( vector<long> { 1, 1 } ),
							ctxtFactory.createCipherText( vector<long> { 1, 1 } ), ctxtFactory.createCipherText( vector<long> { 1, 1 } ) } ) } };

	vector<vector<vector<vector<float>> >> weightsV { vector<vector<vector<float>>> { vector<vector<float>>( 3, vector<float> { .1, .1, .1 } ) } }; //
	vector<float> biasV { 0 };
	vector<vector<vector<vector<float>> >> expectedOutputV { vector<vector<vector<float>>> {
			vector<vector<float>> {
					vector<float> { 0.16000001, 0.36, 0.36, 0.16000001 },
					vector<float> { 0.36, 0.81000006, 0.81000006, 0.36 },
					vector<float> { 0.36, 0.81000006, 0.81000006, 0.36 },
					vector<float> { 0.16000001, 0.36, 0.36, 0.16000001 } } } };

	try {
		HETensorFactory<HELibCipherText> hetfactory( &ctxtFactory );
		PlainTensorFactory<float> ptFactory;

		TensorP<HELibCipherText> input = hetfactory.create( Shape( { 1, 1, 4, 4 } ) );
		input->init( inputVolume );

		TensorP<float> weights = ptFactory.create( Shape( { 1, 1, 3, 3 } ) );

		weights->init( weightsV );
		TensorP<float> biases = ptFactory.create( Shape( { 1 } ) );
		biases->init( biasV );

		Convolution2D<HELibCipherText, float, HETensor<HELibCipherText>, PlainTensor<float> > layer( "test",
				SquareActivation<HELibCipherText>::getSharedPointer(), 1, 3, 1, PADDING_MODE::SAME, input, &hetfactory,
				&ptFactory );

		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );
		//activate layer
		layer.feedForward();

		//create expected tensor
		TensorP<float> expectedOutput = ptFactory.create( Shape( { 1, 1, 4, 4 } ) );
		expectedOutput->init( expectedOutputV );
		TensorP<double> decrypted = ( (HETensor<HELibCipherText>*) layer.output().get() )->decryptDouble();
		cout << decrypted->shape << endl;
		cout << *decrypted << endl;
		return finishTest<double, float>( decrypted, expectedOutput, __func__ );

	} catch ( const std::bad_alloc& e ) {
		cout << "Allocation failed: " << e.what() << endl;
		return false;
	}

}

bool HE_convTest1_ValidBatch_CKKS() {

	cout << "Running " << __func__ << " " << endl;

	HELibCipherTextFactory ctxtFactory( /*L*/4096, /*m*/2 * 4, /*r*/32 );

	vector<vector<vector<vector<HELibCipherText>> >> inputVolume { vector<vector<vector<HELibCipherText>>> {
			vector<vector<HELibCipherText>>( 4,
					vector<HELibCipherText> { ctxtFactory.createCipherText( vector<double> { 0.1, 0.1 } ), ctxtFactory.createCipherText( vector<double> { 0.1, 0.1 } ),
							ctxtFactory.createCipherText( vector<double> { 0.1, 0.1 } ), ctxtFactory.createCipherText( vector<double> { 0.1, 0.1 } ) } ) } };

	vector<vector<vector<vector<float>> >> weightsV { vector<vector<vector<float>>> { vector<vector<float>>( 3, vector<float> { .1, .1, .1 } ) } }; //
	vector<float> biasV { 0.1 };
	vector<vector<vector<vector<float>> >> expectedOutputV( 2, vector<vector<vector<float>>> {
			vector<vector<float>> {
					vector<float> { 0.0361, 0.0361 },
					vector<float> { 0.0361, 0.0361, } } } );

	try {
		HETensorFactory<HELibCipherText> hetfactory( &ctxtFactory );
		PlainTensorFactory<float> ptFactory;

		TensorP<HELibCipherText> input = hetfactory.create( Shape( { 1, 1, 4, 4 } ) );
		input->init( inputVolume );

		TensorP<float> weights = ptFactory.create( Shape( { 1, 1, 3, 3 } ) );

		weights->init( weightsV );
		TensorP<float> biases = ptFactory.create( Shape( { 1 } ) );
		biases->init( biasV );

		Convolution2D<HELibCipherText, float, HETensor<HELibCipherText>, PlainTensor<float> > layer( "test",
				SquareActivation<HELibCipherText>::getSharedPointer(), 1, 3, 1, PADDING_MODE::VALID, input, &hetfactory,
				&ptFactory );

		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );
		//activate layer
		layer.feedForward();

		//create expected tensor
		TensorP<float> expectedOutput = ptFactory.create( Shape( { 2, 1, 2, 2 } ) );
		expectedOutput->init( expectedOutputV );
		TensorP<double> decrypted = ( (HETensor<HELibCipherText>*) layer.output().get() )->decryptDouble();
		cout << decrypted->shape << endl;
		cout << *decrypted << endl;
		return finishTest<double, float>( decrypted, expectedOutput, __func__ );

	} catch ( const std::bad_alloc& e ) {
		cout << "Allocation failed: " << e.what() << endl;
		return false;
	}

}

bool HE_convTest2_Valid_CKKS() {

	cout << "Running " << __func__ << " " << endl;

	HELibCipherTextFactory ctxtFactory( /*L*/4096, /*m*/2 * 4, /*r*/32 );

//	auto inputVolume = range<double>( 3.2, 0.1 );
//	auto weightsV = range<float>( 3.5, 0.1 );
	auto inputVolume = range<double>( 3.2, 0.1 );
	auto weightsV = range<float>( 3.5, 0.1  );
	vector<float> biasV { 0.1, 0.2 };
	vector<vector<vector<vector<float>>>> expectedOutputV( 1, vector<vector<vector<float>>> {
			vector<vector<float>> {
					vector<float> { 28.029999, 29.560001 },
					vector<float> { 34.149998, 35.68 } },
			vector<vector<float>> {
					vector<float> { 70.25, 75.02  },
					vector<float> {  89.329994, 94.1    } } } );

	try {
		HETensorFactory<HELibCipherText> hetfactory( &ctxtFactory );
		PlainTensorFactory<float> ptFactory;

		std::vector<HELibCipherText> cipherTexts;
		for ( auto thing : inputVolume )
			cipherTexts.push_back( ctxtFactory.createCipherText( vector<double>{ thing, thing } ) );
		TensorP<HELibCipherText> input = hetfactory.create( Shape( { 1, 2, 4, 4 } ) );
		input->flatten();
		input->init( cipherTexts );
		input->reshape( Shape( { 1, 2, 4, 4 } ) );

		TensorP<float> weights = ptFactory.create( Shape( { 2, 2, 3, 3 } ) );
		weights->flatten();
		weights->init( weightsV );
		weights->reshape( Shape( { 2, 2, 3, 3 } ) );

		TensorP<float> biases = ptFactory.create( Shape( { 2 } ) );
		biases->init( biasV );

		Convolution2D<HELibCipherText, float, HETensor<HELibCipherText>, PlainTensor<float> > layer( "test",
				LinearActivation<HELibCipherText>::getSharedPointer(), 2, 3, 1, PADDING_MODE::VALID, input, &hetfactory,
				&ptFactory );

		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );
		//activate layer
		layer.feedForward();

		//create expected tensor
		TensorP<float> expectedOutput = ptFactory.create( Shape( { 1, 2, 2, 2 } ) );
		expectedOutput->init( expectedOutputV );
		TensorP<double> decrypted = ( (HETensor<HELibCipherText>*) layer.output().get() )->decryptDouble();
		cout << decrypted->shape << endl;
		cout << *decrypted << endl;
		cout << *expectedOutput << endl;

		cout << "############# try plaintext #############" << endl;
		cout << "#########################################" << endl;
		PlainTensorFactory<double> dtf;

		inputVolume.insert( inputVolume.end(), inputVolume.begin(), inputVolume.end()  );
		TensorP<double> inputPlain = dtf.create( Shape( { 2, 2, 4, 4 } ) );
		inputPlain->flatten();
		inputPlain->init( inputVolume );
		inputPlain->reshape( Shape( { 1, 2, 4, 4 } ) );

		Convolution2D<double, float, PlainTensor<double>, PlainTensor<float> > layerPlain( "test",
				LinearActivation<double>::getSharedPointer(), 2, 3, 1, PADDING_MODE::VALID, inputPlain, &dtf,
				&ptFactory );

		layerPlain.output()->init();
		//set weights/biases of layer
		layerPlain.weights( weights );
		layerPlain.biases( biases );
		//activate layer
		layerPlain.feedForward();

		cout << *layerPlain.output().get() << endl;
		cout << *expectedOutput << endl;

		return finishTest<double, float>( decrypted, expectedOutput, __func__ );

	} catch ( const std::bad_alloc& e ) {
		cout << "Allocation failed: " << e.what() << endl;
		return false;
	}

}



//bool HE_convTest1_validPad() {
//	HELibCipherTextFactory ctxtFactory;
//
//	vector<vector<vector<vector<HELibCipherText>> >> inputVolume { vector<
//			vector<vector<HELibCipherText>>> { vector<vector<HELibCipherText>>(
//			4,
//			vector<HELibCipherText> { ctxtFactory.createCipherText( 1 ),
//					ctxtFactory.createCipherText( 1 ), ctxtFactory
//							.createCipherText( 1 ),
//					ctxtFactory.createCipherText( 1 ) } ) } };
//	vector<vector<vector<vector<long>>>> weightsV {
//			vector<vector<vector<long>>> { vector<vector<long>>( 3,
//					vector<long> { 1, 1, 1 } ) } };
//	vector<long> biasV { 1 };
//	vector<vector<vector<vector<long>>>> expectedOutputV { vector<
//			vector<vector<long>>> { vector<vector<long>> { vector<long> { 100,
//			100 }, vector<long> { 100, 100 } } } };
//
//	try {
//		HETensorFactory<HELibCipherText> hetfactory( &ctxtFactory );
//		PlainTensorFactory<long> ptFactory;
//
//		HETensorP<HELibCipherText> input = hetfactory.create( Shape( { 1, 1, 4,
//				4 } ) );
//		input->init( inputVolume );
//
//		TensorP<long> weights = ptFactory.create( Shape( { 1, 1, 3, 3 } ) );
//
//		weights->init( weightsV );
//		TensorP<long> biases = ptFactory.create( Shape( { 1 } ) );
//		biases->init( biasV );
//
//		Convolution2D<HELibCipherText, long, HETensor<HELibCipherText>,
//				PlainTensor<long> > layer( "test",
//				SquareActivation<HELibCipherText>::getSharedPointer(), 1, 3, 1,
//				PADDING_MODE::SAME, input, &hetfactory, &ptFactory );
//
//		layer.output()->init();
//		//set weights/biases of layer
//		layer.weights( weights );
//		layer.biases( biases );
//		//activate layer
//		layer.feedForward();
//
//		//compare w/ expected
//		if ( ( (HELibCipherTextFactory*) hetfactory.ciphertextFactory() )
//				->useBFV ) {
//			//create expected tensor
//			TensorP<long> expectedOutput = ptFactory.create(
//					Shape( { 1, 1, 4, 4 } ) );
//			expectedOutput->init( expectedOutputV );
//			TensorP<long> decrypted = ( (HETensor<HELibCipherText>*) layer
//					.output().get() )->decryptLong();
//			return finishTest<long, long>( decrypted, expectedOutput, __func__ );
//		} else {
//			//create expected tensor
//			TensorP<long> expectedOutput = ptFactory.create(
//					Shape( { 1, 1, 4, 4 } ) );
//			expectedOutput->init( expectedOutputV );
//			TensorP<double> decrypted = ( (HETensor<HELibCipherText>*) layer
//					.output().get() )->decryptDouble();
//			return finishTest<double, long>( decrypted, expectedOutput,
//					__func__ );
//		}
//
//	} catch ( const std::bad_alloc& e ) {
//		cout << "Allocation failed: " << e.what() << endl;
//		return false;
//	}
//}
//
//bool HE_convTest2_validPad() {
//	cout << "Running " << __func__ << " " << endl;
//	return true;
//	HELibCipherTextFactory ctxtFactory;
//
//	vector<vector<vector<vector<HELibCipherText>> >> inputVolume { vector<
//			vector<vector<HELibCipherText>>> { vector<vector<HELibCipherText>>(
//			4,
//			vector<HELibCipherText> { ctxtFactory.createCipherText( 1 ),
//					ctxtFactory.createCipherText( 2 ), ctxtFactory
//							.createCipherText( 2 ),
//					ctxtFactory.createCipherText( 1 ) } ) } };
//	vector<vector<vector<vector<long>>>> weightsV {
//			vector<vector<vector<long>>> { vector<vector<long>>( 3,
//					vector<long> { 1, 1, 1 } ) } }; //
//	vector<long> biasV { 1 };
//	vector<vector<vector<vector<long>>>> expectedOutputV { vector<
//			vector<vector<long>>> { vector<vector<long>> { vector<long> { 256,
//			256 }, vector<long> { 256, 256 } } } };
//	try {
//		HETensorFactory<HELibCipherText> hetfactory( &ctxtFactory );
//		PlainTensorFactory<long> ptFactory;
//
//		HETensorP<HELibCipherText> input = hetfactory.create( Shape( { 1, 1, 4,
//				4 } ) );
//		input->init( inputVolume );
//
//		TensorP<long> weights = ptFactory.create( Shape( { 1, 1, 3, 3 } ) );
//
//		weights->init( weightsV );
//		TensorP<long> biases = ptFactory.create( Shape( { 1 } ) );
//		biases->init( biasV );
//
//		Convolution2D<HELibCipherText, long, HETensor<HELibCipherText>,
//				PlainTensor<long> > layer( "test",
//				SquareActivation<HELibCipherText>::getSharedPointer(), 1, 3, 1,
//				PADDING_MODE::SAME, input, &hetfactory, &ptFactory );
//
//		layer.output()->init();
//		//set weights/biases of layer
//		layer.weights( weights );
//		layer.biases( biases );
//		//activate layer
//		layer.feedForward();
//
//		//compare w/ expected
//		if ( ( (HELibCipherTextFactory*) hetfactory.ciphertextFactory() )
//				->useBFV ) {
//			//create expected tensor
//			TensorP<long> expectedOutput = ptFactory.create(
//					Shape( { 1, 1, 4, 4 } ) );
//			expectedOutput->init( expectedOutputV );
//			TensorP<long> decrypted = ( (HETensor<HELibCipherText>*) layer
//					.output().get() )->decryptLong();
//			return finishTest<long, long>( decrypted, expectedOutput, __func__ );
//		} else {
//			//create expected tensor
//			TensorP<long> expectedOutput = ptFactory.create(
//					Shape( { 1, 1, 4, 4 } ) );
//			expectedOutput->init( expectedOutputV );
//			TensorP<double> decrypted = ( (HETensor<HELibCipherText>*) layer
//					.output().get() )->decryptDouble();
//			return finishTest<double, long>( decrypted, expectedOutput,
//					__func__ );
//		}
//
//	} catch ( const std::bad_alloc& e ) {
//		cout << "Allocation failed: " << e.what() << endl;
//		return false;
//	}
//}

bool HE_denseTest1() {
	cout << "Running " << __func__ << " " << endl;

	HELibCipherTextFactory ctxtFactory;

	vector<vector<HELibCipherText>> inputVolume { vector<HELibCipherText> { vector<HELibCipherText>( 6, ctxtFactory.createCipherText( 1 ) ) } };

	vector<vector<long>> weightsV( 3, vector<long>( 6, 1 ) );
	vector<long> biasV( 3, 1 );

	vector<vector<long>> expectedOutputV { vector<long>( 3, 49 ) };

	try {
		HETensorFactory<HELibCipherText> hetfactory( &ctxtFactory );
		PlainTensorFactory<long> ptFactory;

		TensorP<HELibCipherText> input = hetfactory.create( Shape( { 1, 6 } ) );
		input->init( inputVolume );

		TensorP<long> weights = ptFactory.create( Shape( { 3, 6 } ) );

		weights->init( weightsV );
		TensorP<long> biases = ptFactory.create( Shape( { 3 } ) );
		biases->init( biasV );

		Dense<HELibCipherText, long, HETensor<HELibCipherText>, PlainTensor<long> > layer( "test",
				SquareActivation<HELibCipherText>::getSharedPointer(), 3, input, &hetfactory,
				&ptFactory );

		layer.output()->init();
		//set weights/biases of layer
		layer.weights( weights );
		layer.biases( biases );
		//activate layer
		layer.feedForward();

		//compare w/ expected
		if ( ( (HELibCipherTextFactory*) hetfactory.ciphertextFactory() )->useBFV ) {
			//create expected tensor
			TensorP<long> expectedOutput = ptFactory.create( Shape( { 1, 3 } ) );
			expectedOutput->init( expectedOutputV );
			TensorP<long> decrypted = ( (HETensor<HELibCipherText>*) layer.output().get() )->decryptLong();
			return finishTest<long, long>( decrypted, expectedOutput, __func__ );
		} else {
			//create expected tensor
			TensorP<long> expectedOutput = ptFactory.create( Shape( { 1, 1, 4, 4 } ) );
			expectedOutput->init( expectedOutputV );
			TensorP<double> decrypted = ( (HETensor<HELibCipherText>*) layer.output().get() )->decryptDouble();
			return finishTest<double, long>( decrypted, expectedOutput, __func__ );
		}

	} catch ( const std::bad_alloc& e ) {
		cout << "Allocation failed: " << e.what() << endl;
		return false;
	}


	return true;
}

bool HE_denseTest2() {
	cout << "Running " << __func__ << " " << endl;
	return true;
}


bool HE_SamePaddFloats() {

	cout << "Running " << __func__ << " " << endl;


	auto conv1OutV = loadConvLayerOutput1<float>( "conv2d_1.txt", 32, 5, 14 * 14, false, "src/experiments/cryptonet/reference_output/" );

	int batchSize = 1;
	HELibCipherTextFactory ctxtFactory( /*L*/4096, /*m*/batchSize * 4, /*r*/32 );
	HETensorFactory<HELibCipherText> hetfactory( &ctxtFactory );
	// create input data
	mnist::MNIST_dataset<uint8_t, uint8_t> dataset = loadMNIST( "src/mnist" );

	TensorP<HELibCipherText> input = hetfactory.create( Shape( { 1, 28 * 28 } ) );

	// TODO this code is a good point to refactor and move it into the CipherTextFactory
	std::vector<std::vector<uint8_t>> X( dataset.test_images.begin(), dataset.test_images.begin() + batchSize );
	std::vector<std::vector<HELibCipherText>> encImages( 1, std::vector<HELibCipherText>() ); /// nested vectors to simulate batches
	for ( size_t i = 0; i < X[ 0 ].size(); i++ ) {
		/// contains every i th from all images in the test test batch
		vector<double> lp;
		/// iterate over all images in the batch and extract ith pixel
		for ( int b = 0; b < batchSize; b++ )
			lp.push_back( dataset.test_images[ b ][ i ] );
		/// create a ciphertext contatining all the ith pixles
		encImages[ 0 ].push_back( ctxtFactory.createCipherText( lp ) );
	}
	// END TODO
	input->init( encImages );
	input->reshape( Shape { 1, 1, 28, 28 } );

	PlainTensorFactory<float> ptFactory;
	Convolution2D<HELibCipherText, float, HETensor<HELibCipherText>, PlainTensor<float> > layer( "conv2d_1",
			SquareActivation<HELibCipherText>::getSharedPointer(), 1, 5, 2, PADDING_MODE::SAME, input, &hetfactory,
			&ptFactory );

	layer.output()->init();
	//set weights/biases of layer
	layer.loadWeights( "src/experiments/cryptonet/savedWeights/" );
	//activate layer

	cout << *layer.weights() << endl;
	cout << *layer.biases() << endl;
	layer.feedForward();

	TensorP<double> decrypted = ( (HETensor<HELibCipherText>*) layer.output().get() )->decryptDouble();
	cout << decrypted->shape << endl;
	cout << *decrypted << endl;
	cout << conv1OutV[ 0 ][ 0 ] << endl;
	return false; //FIXME do actual comparison

}







