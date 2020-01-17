
/*
#include "../../test/TestCommons.h"
 * CompleteNetworkTests.cpp
 *
 *  Created on: Feb 5, 2019
 *      Author: robert
 */


#include "CompleteNetworkTests.h"
#include "../src/architecture/HEBackend/HETensor.h"
#include "../src/architecture/HEBackend/helib/HELIbCipherText.h"

using namespace std;

template<class data_t, class weight_t>
bool completeNetworkTest() {
	cout << "Running " << __func__<<  " " << flush;

	mnist::MNIST_dataset<uint8_t,uint8_t> dataset = loadMNIST("src/mnist");
	PlainTensorFactory<data_t> factory;


//	input shape should be batch, channels, length/width
	TensorP<data_t> input = factory.create( Shape( { 32, 1, 28, 28 } ) );
//	input->init( plainImages );

	//instantiate model
	Model<data_t, weight_t, PlainTensor<data_t>,
			PlainTensor<weight_t> > completeNetwork(
			MemoryUsage::greedy, &factory,
			&factory );

	//conv1
	completeNetwork.addLayer( std::make_shared<
			Convolution2D<data_t, weight_t, PlainTensor<data_t>,
					PlainTensor<weight_t> >>( "conv2d_1",
			SquareActivation<data_t>::getSharedPointer(), 32, 5, 2,
					PADDING_MODE::SAME, input, &factory, &factory ) );
	//conv2
	completeNetwork.addLayer(
			std::make_shared<
					Convolution2D<data_t, weight_t, PlainTensor<data_t>,
							PlainTensor<weight_t> >>( "conv2d_2",
					LinearActivation<data_t>::getSharedPointer(), 64, 5, 2,
					PADDING_MODE::VALID, &factory, &factory ) );

	// flatten
	completeNetwork.addLayer(
			std::make_shared<Flatten<data_t, weight_t, PlainTensor<data_t>, PlainTensor<weight_t> >>( "flatten", &factory,
					&factory ) );
	//dense 1
	completeNetwork.addLayer(
			std::make_shared<
					Dense<data_t, weight_t, PlainTensor<data_t>, PlainTensor<weight_t> >>(
					"dense_1", SquareActivation<data_t>::getSharedPointer(), 100,
					&factory, &factory ) );

	//dense 2
	//TODO: change to and implement softmax/hard activation?
	completeNetwork.addLayer(
			std::make_shared<
					Dense<data_t, weight_t, PlainTensor<data_t>, PlainTensor<weight_t> >>(
					"dense_2", LinearActivation<data_t>::getSharedPointer(), 10,
					&factory, &factory ) );

	completeNetwork.loadWeights( "savedWeights/" );

	// use only 128 instances because it is fast
	std::vector<std::vector<uint8_t>> X( dataset.test_images.begin(), dataset.test_images.begin() + 128 );
	std::vector<uint8_t> Y( dataset.test_labels.begin(), dataset.test_labels.begin() + 128 );

	float acc = completeNetwork.evaluate( X, Y );

	return acc > 0.98;
}
bool completeNetworkTestLong() {
	cout << "Running " << __func__ << " " << flush;
	return completeNetworkTest<long, long>();

}


bool completeNetworkTestFloat(){
	cout << "Running " << __func__<<  " " << flush;
	mnist::MNIST_dataset<uint8_t,uint8_t> dataset = loadMNIST("src/mnist");

	return completeNetworkTest<float, float>();
}



bool completeNetworkTestHELibBFV() {
	typedef HELibCipherText data_t;
	typedef long weight_t;
	cout << "Running " << __func__<< " " << flush;

	mnist::MNIST_dataset<uint8_t,uint8_t> dataset = loadMNIST("src/mnist");
	HELibCipherTextFactory ctxtFactory;

	HETensorFactory<HELibCipherText> hetfactory( &ctxtFactory );
	PlainTensorFactory<long> ptFactory;

	uint batchSize = ctxtFactory.batchsize();


	//	input shape should be batch, channels, length/width
	TensorP<data_t> input = hetfactory.create( Shape( { 1, 1, 28, 28 } ) );


	//instantiate model
	Model<data_t, weight_t, HETensor<data_t>,
	PlainTensor<weight_t> > completeNetwork(
			MemoryUsage::greedy, &hetfactory,
			&ptFactory );

	//conv1
	completeNetwork.addLayer( std::make_shared<
			Convolution2D<data_t, weight_t, HETensor<data_t>, PlainTensor<weight_t> >>( "conv2d_1",
					SquareActivation<data_t>::getSharedPointer(), 32, 5, 2,
					PADDING_MODE::SAME, input, &hetfactory,
					&ptFactory ) );
	//conv2
	completeNetwork.addLayer(
			std::make_shared<
			Convolution2D<data_t, weight_t, HETensor<data_t>, PlainTensor<weight_t> >>( "conv2d_2",
					LinearActivation<data_t>::getSharedPointer(), 64, 5, 2,
					PADDING_MODE::VALID, &hetfactory,
					&ptFactory ) );

	// flatten
	completeNetwork.addLayer(
			std::make_shared<Flatten<data_t, weight_t, HETensor<data_t>, PlainTensor<weight_t> >>( "flatten",
					&hetfactory,
					&ptFactory ) );
	//dense 1
	completeNetwork.addLayer(
			std::make_shared<
			Dense<data_t, weight_t, HETensor<data_t>, PlainTensor<weight_t> >>(
					"dense_1", SquareActivation<data_t>::getSharedPointer(), 100,
					&hetfactory, &ptFactory ) );

	//dense 2
	//TODO: change to and implement softmax/hard activation?
	completeNetwork.addLayer(
			std::make_shared<
			Dense<data_t, weight_t, HETensor<data_t>, PlainTensor<weight_t> >>(
					"dense_2", LinearActivation<data_t>::getSharedPointer(), 10,
					&hetfactory, &ptFactory ) );

	cout << "Here" << endl;
	completeNetwork.loadWeights( "savedWeights/" );
	std::cout << "batchsize: " << batchSize << std::endl;

	// use only 100 instances because it is fast
	std::vector<std::vector<uint8_t>> X( dataset.test_images.begin(), dataset.test_images.begin() + batchSize );
	std::vector<uint8_t> Y( dataset.test_labels.begin(), dataset.test_labels.begin() + batchSize );

	// encrypt the testdata
	std::vector<std::vector<HELibCipherText>> encImages( 1, std::vector<HELibCipherText>() );
	for ( size_t i = 0; i < X [ 0 ].size(); i++ ) {
		vector<long> lp;
		for ( uint b = 0; b < batchSize; b++ ) {
			lp.push_back( unsigned( dataset.test_images [ b ] [ i ] ) );
		}
		auto enc = ctxtFactory.createCipherText( lp );
		encImages [ 0 ].push_back( enc );
	}

	try {
	completeNetwork.evaluate<HELibCipherText, uint8_t>( encImages, Y );
	} catch ( ... ) {

	}
	std::cout << "batchsize: " << batchSize << std::endl;
	if ( ( (HELibCipherTextFactory*) hetfactory.ciphertextFactory() )->useBFV ) {
		//create expected tensor
		TensorP<long> decrypted = ( (HETensor<HELibCipherText>*) completeNetwork.mLayers.back()->output().get() )->decryptLong();
		auto preds = decrypted->argmaxVector();
		float acc = completeNetwork.accuracy( preds, Y );
		std::cout << preds << std::endl;
		std::cout << "[";
		for ( auto p : Y )
			std::cout << (unsigned) p << " ";
		std::cout << "]" << std::endl;
		cout << "Accuracy: " << acc << std::endl;
		return acc < 0.9;

	}
	return false;
}

bool completeNetworkTestHELibCKKS() {
	typedef HELibCipherText data_t;
	typedef long weight_t;
	cout << "Running " << __func__ << " " << flush;

	mnist::MNIST_dataset<uint8_t, uint8_t> dataset = loadMNIST( "src/mnist" );
	HELibCipherTextFactory ctxtFactory( 0, false );

	HETensorFactory<HELibCipherText> hetfactory( &ctxtFactory );
	PlainTensorFactory<long> ptFactory;

	uint batchSize = ctxtFactory.batchsize();

	//	input shape should be batch, channels, length/width
	TensorP<data_t> input = hetfactory.create( Shape( { 1, 1, 28, 28 } ) );

	//instantiate model
	Model<data_t, weight_t, HETensor<data_t>, PlainTensor<weight_t> > completeNetwork( MemoryUsage::greedy, &hetfactory,
			&ptFactory );

	//conv1
	completeNetwork.addLayer(
			std::make_shared<Convolution2D<data_t, weight_t, HETensor<data_t>, PlainTensor<weight_t> >>( "conv2d_1",
					SquareActivation<data_t>::getSharedPointer(), 32, 5, 2, PADDING_MODE::SAME, input, &hetfactory,
					&ptFactory ) );
	//conv2
	completeNetwork.addLayer(
			std::make_shared<Convolution2D<data_t, weight_t, HETensor<data_t>, PlainTensor<weight_t> >>( "conv2d_2",
					LinearActivation<data_t>::getSharedPointer(), 64, 5, 2, PADDING_MODE::VALID, &hetfactory,
					&ptFactory ) );

	// flatten
	completeNetwork.addLayer(
			std::make_shared<Flatten<data_t, weight_t, HETensor<data_t>, PlainTensor<weight_t> >>( "flatten",
					&hetfactory, &ptFactory ) );
	//dense 1
	completeNetwork.addLayer(
			std::make_shared<Dense<data_t, weight_t, HETensor<data_t>, PlainTensor<weight_t> >>( "dense_1",
					SquareActivation<data_t>::getSharedPointer(), 100, &hetfactory, &ptFactory ) );

	//dense 2
	//TODO: change to and implement softmax/hard activation?
	completeNetwork.addLayer(
			std::make_shared<Dense<data_t, weight_t, HETensor<data_t>, PlainTensor<weight_t> >>( "dense_2",
					LinearActivation<data_t>::getSharedPointer(), 10, &hetfactory, &ptFactory ) );

	cout << "Here" << endl;
	completeNetwork.loadWeights( "savedWeights/" );
	std::cout << "batchsize: " << batchSize << std::endl;

	// use only 100 instances because it is fast
	std::vector<std::vector<uint8_t>> X( dataset.test_images.begin(), dataset.test_images.begin() + batchSize );
	std::vector<uint8_t> Y( dataset.test_labels.begin(), dataset.test_labels.begin() + batchSize );

	// encrypt the testdata
	std::vector<std::vector<HELibCipherText>> encImages( 1, std::vector<HELibCipherText>() );
	for ( size_t i = 0; i < X [ 0 ].size(); i++ ) {
		vector<long> lp;
		for ( uint b = 0; b < batchSize; b++ ) {
			lp.push_back( unsigned( dataset.test_images [ b ] [ i ] ) );
		}
		auto enc = ctxtFactory.createCipherText( lp );
		encImages [ 0 ].push_back( enc );
	}

	try {
		completeNetwork.evaluate<HELibCipherText, uint8_t>( encImages, Y );
	} catch ( ... ) {

	}
	std::cout << "batchsize: " << batchSize << std::endl;
	//create expected tensor
	TensorP<double> decrypted = ( (HETensor<HELibCipherText>*) completeNetwork.mLayers.back()->output().get() )
			->decryptDouble();
	auto preds = decrypted->argmaxVector();
	float acc = completeNetwork.accuracy( preds, Y );
	std::cout << preds << std::endl;
	std::cout << "[";
	for ( auto p : Y )
		std::cout << (unsigned) p << " ";
	std::cout << "]" << std::endl;
	cout << "Accuracy: " << acc << std::endl;
	return acc > 0.9;

}

template<class data_t, class weight_t>
bool compareLayerByLayer() {
	cout << "Running " << __func__ << " " << flush;

	mnist::MNIST_dataset<uint8_t, uint8_t> dataset = loadMNIST( "src/mnist" );
	PlainTensorFactory<data_t> factory;


//	input shape should be batch, channels, length/width
	TensorP<data_t> input = factory.create( Shape( { 32, 1, 28, 28 } ) );
//	input->init( plainImages );

	//instantiate model
	Model<data_t, weight_t, PlainTensor<data_t>, PlainTensor<weight_t> > completeNetwork( MemoryUsage::greedy, &factory,
			&factory );

	//conv1
	completeNetwork.addLayer(
			std::make_shared<Convolution2D<data_t, weight_t, PlainTensor<data_t>, PlainTensor<weight_t> >>( "conv2d_1",
					SquareActivation<data_t>::getSharedPointer(), 32, 5, 2, PADDING_MODE::SAME, input, &factory,
					&factory ) );
	//conv2
	completeNetwork.addLayer(
			std::make_shared<Convolution2D<data_t, weight_t, PlainTensor<data_t>, PlainTensor<weight_t> >>( "conv2d_2",
					LinearActivation<data_t>::getSharedPointer(), 64, 5, 2, PADDING_MODE::VALID, &factory, &factory ) );

	// flatten
	completeNetwork.addLayer(
			std::make_shared<Flatten<data_t, weight_t, PlainTensor<data_t>, PlainTensor<weight_t> >>( "flatten",
					&factory, &factory ) );
	//dense 1
	completeNetwork.addLayer(
			std::make_shared<Dense<data_t, weight_t, PlainTensor<data_t>, PlainTensor<weight_t> >>( "dense_1",
					SquareActivation<data_t>::getSharedPointer(), 100, &factory, &factory ) );

	//dense 2
	//TODO: change to and implement softmax/hard activation?
	completeNetwork.addLayer(
			std::make_shared<Dense<data_t, weight_t, PlainTensor<data_t>, PlainTensor<weight_t> >>( "dense_2",
					LinearActivation<data_t>::getSharedPointer(), 10, &factory, &factory ) );

	completeNetwork.loadWeights( "savedWeights/" );

	std::vector<TensorP<data_t>> referenceOutPuts;

	auto conv1OutV = loadConvLayerOutput1<data_t>( "conv2d_1.txt", 32, 32, 14 * 14 );
	TensorP<data_t> conv1Out = factory.create( Shape( { 32, 32, 14 * 14 } ) );
	conv1Out->init( conv1OutV );
	conv1Out->reshape( Shape( { 32, 32, 14, 14 } ) );
	referenceOutPuts.push_back( std::move( conv1Out ) );

	auto conv2OutV = loadConvLayerOutput1<data_t>( "conv2d_2.txt", 32, 64, 25 );
	TensorP<data_t> conv2Out = factory.create( Shape( { 32, 64, 25 } ) );
	conv2Out->init( conv2OutV );
	conv2Out->reshape( Shape( { 32, 64, 5, 5 } ) );
	referenceOutPuts.push_back( std::move( conv2Out ) );

	auto flattenOutV = loadDenseLayerOutput1<data_t>( "flatten_1.txt", 32, 1600 );
	TensorP<data_t> flattenOut = factory.create( Shape( { 32, 1600 } ) );
	flattenOut->init( flattenOutV );
	flattenOut->reshape( Shape( { 32, 1600 } ) );
	referenceOutPuts.push_back( std::move( flattenOut ) );

	auto dense1OutV = loadDenseLayerOutput1<data_t>( "dense_1.txt", 32, 100 );
	TensorP<data_t> dense1Out = factory.create( Shape( { 32, 100 } ) );
	dense1Out->init( dense1OutV );
	dense1Out->reshape( Shape( { 32, 100 } ) );
	referenceOutPuts.push_back( std::move( dense1Out ) );

	auto dense2OutV = loadDenseLayerOutput1<data_t>( "dense_2.txt", 32, 10 );
	TensorP<data_t> condense2Out = factory.create( Shape( { 32, 10 } ) );
	condense2Out->init( dense2OutV );
	condense2Out->reshape( Shape( { 32, 10 } ) );
	referenceOutPuts.push_back( std::move( condense2Out ) );


	// use only 32 instances because it is fast and we have layer output data for it
	std::vector<std::vector<uint8_t>> X( dataset.test_images.begin(), dataset.test_images.begin() + 32 );
	std::vector<uint8_t> Y( dataset.test_labels.begin(), dataset.test_labels.begin() + 32 );

	completeNetwork.feedInputTensor( X, Y );
	for ( uint i = 0; i < completeNetwork.mLayers.size(); ++i ) {
		completeNetwork.mLayers [ i ]->feedForward();
		TensorP<data_t> refOut = referenceOutPuts [ i ];
		finishTest( completeNetwork.mLayers [ i ]->output(), refOut, completeNetwork.mLayers [ i ]->name() );
	}

	std::cout << completeNetwork.classification() << std::endl;
	std::cout << "[";
	for ( auto p : Y )
		std::cout << (unsigned) p << " ";
	std::cout << "]" << std::endl;
	std::cout << completeNetwork.accuracy( Y ) << std::endl;

	return 0.9 > completeNetwork.accuracy( Y );
}

bool compareLayerByLayerLong() {
	return compareLayerByLayer<long, long>();
}

bool compareLayerByLayerFloat() {
	return compareLayerByLayer<float, float>();
}


template<class data_t, class weight_t>
bool compareLayerByLayerEncrypted() {
	cout << "Running " << __func__ << " " << flush;
	/// load the data set
	mnist::MNIST_dataset<uint8_t, uint8_t> dataset = loadMNIST( "src/mnist" );
	int batchSize = 32;
	HELibCipherTextFactory ctxtFactory( /*L*/8192, /*m*/batchSize * 4, /*r*/32 ); // batchsize needs to be fixed at 32 for this


	HETensorFactory<HELibCipherText> hetfactory( &ctxtFactory );
	PlainTensorFactory<weight_t> ptFactory;

	/// crop the testdata to fit our batchsize
	std::vector<std::vector<uint8_t>> X( dataset.test_images.begin(), dataset.test_images.begin() + batchSize );
	std::vector<uint8_t> Y( dataset.test_labels.begin(), dataset.test_labels.begin() + batchSize );

	std::vector<TensorP<data_t>> referenceOutPuts;

	auto conv1OutV = loadConvLayerOutput1<data_t>( "conv2d_1.txt", batchSize, 5, 14 * 14, false,
			"src/experiments/cryptonet/reference_output/" );
	TensorP<data_t> conv1Out = ptFactory.create( Shape( { batchSize, 5, 14 * 14 } ) );
	conv1Out->init( conv1OutV );
	conv1Out->reshape( Shape( { batchSize, 5, 14, 14 } ) );
	referenceOutPuts.push_back(  conv1Out  );

	auto conv2OutV = loadConvLayerOutput1<data_t>( "conv2d_2.txt", batchSize, 10, 25, false,
			"src/experiments/cryptonet/reference_output/" );
	TensorP<data_t> conv2Out = ptFactory.create( Shape( { batchSize, 10, 25 } ) );
	conv2Out->init( conv2OutV );
	conv2Out->reshape( Shape( { batchSize, 10, 5, 5 } ) );
	referenceOutPuts.push_back(  conv2Out );

	auto flattenOutV = loadDenseLayerOutput1<data_t>( "flatten_1.txt", batchSize, 250, false,
			"src/experiments/cryptonet/reference_output/" );
	TensorP<data_t> flattenOut = ptFactory.create( Shape( { batchSize, 250 } ) );
	flattenOut->init( flattenOutV );
	flattenOut->reshape( Shape( { batchSize, 250 } ) );
	referenceOutPuts.push_back( flattenOut  );

	auto dense1OutV = loadDenseLayerOutput1<data_t>( "dense_1.txt", batchSize, 100, false,
			"src/experiments/cryptonet/reference_output/" );
	TensorP<data_t> dense1Out = ptFactory.create( Shape( { batchSize, 100 } ) );
	dense1Out->init( dense1OutV );
	dense1Out->reshape( Shape( { batchSize, 100 } ) );
	referenceOutPuts.push_back(  dense1Out  );

	auto dense2OutV = loadDenseLayerOutput1<data_t>( "dense_2.txt", batchSize, 10, false,
			"src/experiments/cryptonet/reference_output/" );
	TensorP<data_t> condense2Out = ptFactory.create( Shape( { batchSize, 10 } ) );
	condense2Out->init( dense2OutV );
	condense2Out->reshape( Shape( { batchSize, 10 } ) );
	referenceOutPuts.push_back(  condense2Out  );

//	uint batchSize = ctxtFactory.batchsize();
	TensorP<HELibCipherText> input = hetfactory.create( Shape( { 1, 1, 28, 28 } ) );

	Model<HELibCipherText, weight_t, HETensor<HELibCipherText>, PlainTensor<weight_t> > completeNetwork(
			MemoryUsage::greedy,
			&hetfactory, &ptFactory );

	completeNetwork.addLayer(
			std::make_shared<Convolution2D<HELibCipherText, weight_t, HETensor<HELibCipherText>, PlainTensor<weight_t> >>(
					"conv2d_1", SquareActivation<HELibCipherText>::getSharedPointer(), 5, 5, 2, PADDING_MODE::SAME,
					input, &hetfactory, &ptFactory ) );

	//conv2
	completeNetwork.addLayer(
			std::make_shared<Convolution2D<HELibCipherText, weight_t, HETensor<HELibCipherText>, PlainTensor<weight_t> >>(
					"conv2d_2", SquareActivation<HELibCipherText>::getSharedPointer(), 10, 5, 2, PADDING_MODE::VALID,
					&hetfactory, &ptFactory ) );
	// flatten
	completeNetwork.addLayer(
			std::make_shared<Flatten<HELibCipherText, weight_t, HETensor<HELibCipherText>, PlainTensor<weight_t> >>(
					"flatten", &hetfactory, &ptFactory ) );
	//dense 1
	completeNetwork.addLayer(
			std::make_shared<Dense<HELibCipherText, weight_t, HETensor<HELibCipherText>, PlainTensor<weight_t> >>(
					"dense_1",
					SquareActivation<HELibCipherText>::getSharedPointer(), 100, &hetfactory, &ptFactory ) );

	//dense 2
	completeNetwork.addLayer(
			std::make_shared<Dense<HELibCipherText, weight_t, HETensor<HELibCipherText>, PlainTensor<weight_t> >>(
					"dense_2",
					LinearActivation<HELibCipherText>::getSharedPointer(), 10, &hetfactory, &ptFactory ) );

	completeNetwork.loadWeights( "src/experiments/cryptonet/savedWeights/" );
	std::cout << "batchsize: " << batchSize << std::endl;

//	std::cout<< (*completeNetwork.mLayers [ 1 ]->weights()).shape << " " << std::endl;
//	for( int i = 0; i < 5; i++ )
//		for( int j = 0; j < 5; j++ )
//			std::cout<< (*completeNetwork.mLayers [ 1 ]->weights())[ { 1,1,i,j } ] << " " ;

	/// crop the testdata to fit our batchsize

	/// encrypt the test data
	/// The "shape" of the data as it is loaded is [ instances, pixels ] but we need to reshape it to [ pixels, instances ]
	/// so that we can encrypt all pixels with the same coordinates from multiple images into one ciphertext.
	std::vector<std::vector<HELibCipherText>> encImages( 1, std::vector<HELibCipherText>() ); /// nested vectors to simulate batches
	for ( size_t i = 0; i < X [ 0 ].size(); i++ ) {
		/// contains every i th from all images in the test test batch
		vector<double> lp;
		/// iterate over all images in the batch and extract ith pixel
		for ( int b = 0; b < batchSize; b++ ) {
			lp.push_back( dataset.test_images [ b ] [ i ] );
		}
		/// create a ciphertext contatining all the ith pixles
		auto enc = ctxtFactory.createCipherText( lp );
		encImages [ 0 ].push_back( enc );
	}




	completeNetwork.feedInputTensor( encImages, Y );
	TensorP<double> decryptedInput = ( (HETensor<HELibCipherText>*) completeNetwork.mLayers[ 0 ]->input().get() ) // @suppress("Invalid arguments")
			->decryptDouble();
//	std::cout << *decryptedInput << std::endl;
	for ( uint i = 0; i < completeNetwork.mLayers.size(); ++i ) {
		completeNetwork.mLayers [ i ]->feedForward();
		TensorP<data_t> refOut = referenceOutPuts [ i ];
		TensorP<double> decrypted = ( (HETensor<HELibCipherText>*) completeNetwork.mLayers [ i ]->output().get() ) // @suppress("Invalid arguments")
				->decryptDouble();

		finishTest( decrypted, refOut, completeNetwork.mLayers [ i ]->name() );
		//std::cout << *decrypted << std::endl;
		//std::cout << *refOut << std::endl;
	}

	// run comparison on plaintext

	std::cout << "PLAINTEXTEXTTT" << std::endl;

	TensorP<data_t> inputplain = ptFactory.create( Shape( { batchSize, 1, 28, 28 } ) );

	Model<weight_t, weight_t, PlainTensor<weight_t>, PlainTensor<weight_t> > plainNet(
			MemoryUsage::greedy,
			&ptFactory, &ptFactory );

	plainNet.addLayer(
			std::make_shared<Convolution2D<weight_t, weight_t, PlainTensor<weight_t>, PlainTensor<weight_t> >>(
					"conv2d_1", SquareActivation<weight_t>::getSharedPointer(), 5, 5, 2, PADDING_MODE::SAME,
					inputplain, &ptFactory, &ptFactory ) );

	//conv2
	plainNet.addLayer(
			std::make_shared<Convolution2D<weight_t, weight_t, PlainTensor<weight_t>, PlainTensor<weight_t> >>(
					"conv2d_2", SquareActivation<weight_t>::getSharedPointer(), 10, 5, 2, PADDING_MODE::VALID, &ptFactory, &ptFactory ) );
	// flatten
	plainNet.addLayer(
			std::make_shared<Flatten<weight_t, weight_t, PlainTensor<weight_t>, PlainTensor<weight_t> >>(
					"flatten", &ptFactory, &ptFactory ) );
	//dense 1
	plainNet.addLayer(
			std::make_shared<Dense<weight_t, weight_t, PlainTensor<weight_t>, PlainTensor<weight_t> >>(
					"dense_1",
					SquareActivation<weight_t>::getSharedPointer(), 100, &ptFactory, &ptFactory ) );

	//dense 2
	plainNet.addLayer(
			std::make_shared<Dense<weight_t, weight_t, PlainTensor<weight_t>, PlainTensor<weight_t> >>(
					"dense_2",
					LinearActivation<weight_t>::getSharedPointer(), 10, &ptFactory, &ptFactory ) );


	plainNet.loadWeights( "src/experiments/cryptonet/savedWeights/" );

	plainNet.feedInputTensor( X, Y );
	for ( uint i = 0; i < plainNet.mLayers.size(); ++i ) {
		plainNet.mLayers [ i ]->feedForward();
		TensorP<data_t> refOut = referenceOutPuts [ i ];
		if( i==2 ){
			std::cout <<  *plainNet.mLayers [ i ]->output() << std::endl;
			std::cout <<  *refOut << std::endl;
		}
		finishTest( plainNet.mLayers [ i ]->output(), refOut, plainNet.mLayers [ i ]->name() );
	}

	std::cout << plainNet.classification() << std::endl;
	std::cout << "[";
	for ( auto p : Y )
		std::cout << (unsigned) p << " ";
	std::cout << "]" << std::endl;
	std::cout << plainNet.accuracy( Y ) << std::endl;

	std::cout << "batchsize: " << batchSize << std::endl;
	//create expected tensor
	TensorP<double> decrypted = ( (HETensor<HELibCipherText>*) completeNetwork.mLayers.back()->output().get() )
			->decryptDouble();
	auto preds = decrypted->argmaxVector();
	float acc = completeNetwork.accuracy( preds, Y );
	std::cout << preds << std::endl;
	std::cout << "[";
	for ( auto p : Y )
		std::cout << (unsigned) p << " ";
	std::cout << "]" << std::endl;
	cout << "Accuracy: " << acc << std::endl;
	return acc > 0.9;
}

bool compareLayerByLayerEncryptedFloat() {
	return compareLayerByLayerEncrypted<float, float>();
}

bool completeNetworkTestHELibCKKSFloatWeights() {
	cout << "Running " << __func__ << " " << flush;

	mnist::MNIST_dataset<uint8_t, uint8_t> dataset = loadMNIST( "src/mnist" );
	HELibCipherTextFactory ctxtFactory( 1024, 32 * 4, 8 );


	HETensorFactory<HELibCipherText> hetfactory( &ctxtFactory );
	PlainTensorFactory<float> ptFactory;

	uint batchSize = ctxtFactory.batchsize();

	//	input shape should be batch, channels, length/width
	TensorP<HELibCipherText> input = hetfactory.create( Shape( { 1, 1, 28, 28 } ) );

	//instantiate model
	Model<HELibCipherText, float, HETensor<HELibCipherText>, PlainTensor<float> > completeNetwork( MemoryUsage::greedy, &hetfactory,
			&ptFactory );

	//conv1
	completeNetwork.addLayer(
			std::make_shared<Convolution2D<HELibCipherText, float, HETensor<HELibCipherText>, PlainTensor<float> >>( "conv2d_1",
					SquareActivation<HELibCipherText>::getSharedPointer(), 32, 5, 2, PADDING_MODE::SAME, input, &hetfactory,
					&ptFactory ) );
	//conv2
	completeNetwork.addLayer(
			std::make_shared<Convolution2D<HELibCipherText, float, HETensor<HELibCipherText>, PlainTensor<float> >>( "conv2d_2",
					LinearActivation<HELibCipherText>::getSharedPointer(), 64, 5, 2, PADDING_MODE::VALID, &hetfactory,
					&ptFactory ) );

	// flatten
	completeNetwork.addLayer(
			std::make_shared<Flatten<HELibCipherText, float, HETensor<HELibCipherText>, PlainTensor<float> >>( "flatten",
					&hetfactory, &ptFactory ) );
	//dense 1
	completeNetwork.addLayer(
			std::make_shared<Dense<HELibCipherText, float, HETensor<HELibCipherText>, PlainTensor<float> >>( "dense_1",
					SquareActivation<HELibCipherText>::getSharedPointer(), 100, &hetfactory, &ptFactory ) );

	//dense 2
	completeNetwork.addLayer(
			std::make_shared<Dense<HELibCipherText, float, HETensor<HELibCipherText>, PlainTensor<float> >>( "dense_2",
					LinearActivation<HELibCipherText>::getSharedPointer(), 10, &hetfactory, &ptFactory ) );

	completeNetwork.loadWeights( "src/keras/data/float_saved_weights/" );
	std::cout << "batchsize: " << batchSize << std::endl;

	// use only 100 instances because it is fast
	std::vector<std::vector<uint8_t>> X( dataset.test_images.begin(), dataset.test_images.begin() + batchSize );
	std::vector<uint8_t> Y( dataset.test_labels.begin(), dataset.test_labels.begin() + batchSize );

	// encrypt the testdata
	std::vector<std::vector<HELibCipherText>> encImages( 1, std::vector<HELibCipherText>() );
	for ( size_t i = 0; i < X[ 0 ].size(); i++ ) {
		vector<long> lp;
		for ( uint b = 0; b < batchSize; b++ ) {
			lp.push_back( unsigned( dataset.test_images[ b ][ i ] ) );
		}
		auto enc = ctxtFactory.createCipherText( lp );
		encImages[ 0 ].push_back( enc );
	}

	try {
		completeNetwork.evaluate<HELibCipherText, uint8_t>( encImages, Y );
	} catch ( ... ) {

	}
	std::cout << "batchsize: " << batchSize << std::endl;
	//create expected tensor
	TensorP<double> decrypted = ( (HETensor<HELibCipherText>*) completeNetwork.mLayers.back()->output().get() )
			->decryptDouble();
	auto preds = decrypted->argmaxVector();
	float acc = completeNetwork.accuracy( preds, Y );
	std::cout << preds << std::endl;
	std::cout << "[";
	for ( auto p : Y )
		std::cout << (unsigned) p << " ";
	std::cout << "]" << std::endl;
	cout << "Accuracy: " << acc << std::endl;
	return acc > 0.9;

}

