/*
 * mnist_rnn.cpp
 *
 *  Created on: October 1, 2019
 *      Author: robert
 */


#include "../../src/data/DatasetOperations.h"
#include "../../src/architecture/ActivationFunction.h"
#include "../../src/architecture/Layer.h"
#include "../../src/architecture/PlainTensor.h"
#include "../../src/architecture/Model.h"
#include "../../src/architecture/HEBackend/HETensor.h"
#include "../../src/architecture/HEBackend/helib/HELIbCipherText.h"


int main(int argc, char **argv) {
	/// load the data set
	mnist::MNIST_dataset<uint8_t, uint8_t> dataset = loadMNIST( );
	std::vector<int> batchSizes{ 1, 4, 128, 512, 1024, 2048, 4096, 8192, 1000 };
	for(auto b : batchSizes ){

		int batchSize = b;

		std::cout << "Parameters: " << std::endl;
		std::cout << "L: " <<  4096  << " m: " << batchSize*4 << " r: " << 16 << " (batchsize: " << batchSize << ")" <<  std::endl;

		HELibCipherTextFactory ctxtFactory( 4096, batchSize * 4, 16 ); // L, m,  r

		/// create Tensor factories for both weights and data
		HETensorFactory<HELibCipherText> hetfactory( &ctxtFactory );
		PlainTensorFactory<float> ptFactory;
		TensorP<HELibCipherText> input = hetfactory.create( Shape( { 1, 28, 28 } ) );

		/// instantiate model
		/// we need to specify the way memory is allocated and pass factories for the data tensors and weight tensors
		Model<HELibCipherText, float, HETensor<HELibCipherText>, PlainTensor<float>> completeNetwork( MemoryUsage::greedy, &hetfactory, &ptFactory );

		auto act = PolynomialActivationDegree3<HELibCipherText>::getSharedPointer( -0.00163574303018748, 0, 0.249476365628036, 0, input);

		completeNetwork.addLayer(
				std::make_shared<RNN<HELibCipherText, float, HETensor<HELibCipherText>, PlainTensor<float> >>(
						"simple_rnn_1", act, 128, false,
						input, &hetfactory,
						&ptFactory ) );
		//dense 1
		completeNetwork.addLayer(
				std::make_shared<Dense<HELibCipherText, float, HETensor<HELibCipherText>, PlainTensor<float> >>( "dense_1",
						LinearActivation<HELibCipherText>::getSharedPointer(), 10, &hetfactory, &ptFactory ) );

		/// since we have named the layers the same way they were named in keras we only need to specify the directory that
		/// the exported weights
		completeNetwork.loadWeights( "examples/mnist_rnn/weights/" );
		std::cout << "batchsize: " << batchSize << std::endl;

		/// crop the testdata to fit our batchsize
		std::vector<std::vector<uint8_t>> X( dataset.test_images.begin(), dataset.test_images.begin() + batchSize );
		std::vector<uint8_t> Y( dataset.test_labels.begin(), dataset.test_labels.begin() + batchSize );

		/// encrypt the test data
		/// The "shape" of the data as it is loaded is [ instances, pixels ] but we need to reshape it to [ pixels, instances ]
		/// so that we can encrypt all pixels with the same coordinates from multiple images into one ciphertext.
		std::vector<HELibCipherText> encImages; /// nested vectors to simulate batches
		for ( size_t i = 0; i < X [ 0 ].size(); i++ ) {
			/// contains every i th from all images in the test test batch
			std::vector<double> lp;
			/// iterate over all images in the batch and extract ith pixel
			for ( int b = 0; b < batchSize; b++ ) {
				lp.push_back( unsigned( dataset.test_images [ b ] [ i ] ) / 255. );
			}
			/// create a ciphertext containing all the ith pixels
			auto enc = ctxtFactory.createCipherText( lp );
			encImages.push_back( enc );
		}


		auto orgShape = input->shape;
		input->flatten();
		input->init( encImages );
		input->reshape( orgShape );

		// run the network
		completeNetwork.run();

		/// get the output of the network and decrypt it
		TensorP<double> decrypted = ( (HETensor<HELibCipherText>*) completeNetwork.mLayers.back()->output().get() )->decryptDouble();
		auto preds = decrypted->argmaxVector();
		float acc = completeNetwork.accuracy( preds, Y );
		std::cout << "Accuracy: " << acc << std::endl;
		std::cout << "##############################" << acc << std::endl;
	}
}

