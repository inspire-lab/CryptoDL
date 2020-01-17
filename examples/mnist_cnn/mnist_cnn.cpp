#include "../../src/data/DatasetOperations.h"
#include "../../src/architecture/ActivationFunction.h"
#include "../../src/architecture/Layer.h"
#include "../../src/architecture/PlainTensor.h"
#include "../../src/architecture/Model.h"
#include "../../src/architecture/HEBackend/HETensor.h"
#include "../../src/architecture/HEBackend/helib/HELIbCipherText.h"
#include "../../src/tools/Config.h"
#include "../../src/tools/SystemTools.h"
using namespace std;



/**
 * In this example we are running a convolutional neural network on encrypted data. The network performs
 * classification on the MNIST dataset of hand written digits. The model was originally trained in keras
 * and the weights have been exported using `src/keras/exportWeights.py`. Below is the summary and the
 * configuration of the model. Inside the main function we are redefining the model and feeding it
 * encrypted instances.
 *
 * The code asssumes to be executed from the project root.
 *
 *
 * This is the network we are working with:
 *
 * Layer (type)                 Output Shape              Param #
 * =================================================================
 * conv2d_1 (Conv2D)            (None, 32, 14, 14)        832
 * _________________________________________________________________
 * conv2d_2 (Conv2D)            (None, 64, 5, 5)          51264
 * _________________________________________________________________
 * flatten_1 (Flatten)          (None, 1600)              0
 * _________________________________________________________________
 * dropout_1 (Dropout)          (None, 1600)              0
 * _________________________________________________________________
 * dense_1 (Dense)              (None, 100)               160100
 * _________________________________________________________________
 * dense_2 (Dense)              (None, 10)                1010
 * =================================================================
 *
 * Layer config:
 * {'name': 'conv2d_1', 'trainable': True, 'batch_input_shape': (None, 1, 28, 28), 'dtype': 'float32', 'filters': 32, 'kernel_size': (5, 5), 'strides': (2, 2), 'padding': 'same', 'data_format': 'channels_first', 'dilation_rate': (1, 1), 'activation': 'square', 'use_bias': True, 'kernel_initializer': {'class_name': 'VarianceScaling', 'config': {'scale': 1.0, 'mode': 'fan_avg', 'distribution': 'uniform', 'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}
 * {'name': 'conv2d_2', 'trainable': True, 'filters': 64, 'kernel_size': (5, 5), 'strides': (2, 2), 'padding': 'valid', 'data_format': 'channels_first', 'dilation_rate': (1, 1), 'activation': 'linear', 'use_bias': True, 'kernel_initializer': {'class_name': 'VarianceScaling', 'config': {'scale': 1.0, 'mode': 'fan_avg', 'distribution': 'uniform', 'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}
 * {'name': 'flatten_1', 'trainable': True}
 * {'name': 'dropout_1', 'trainable': True, 'rate': 0.5, 'noise_shape': None, 'seed': None}
 * {'name': 'dense_1', 'trainable': True, 'units': 100, 'activation': 'square', 'use_bias': True, 'kernel_initializer': {'class_name': 'VarianceScaling', 'config': {'scale': 1.0, 'mode': 'fan_avg', 'distribution': 'uniform', 'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}
 * {'name': 'dense_2', 'trainable': True, 'units': 10, 'activation': 'softmax', 'use_bias': True, 'kernel_initializer': {'class_name': 'VarianceScaling', 'config': {'scale': 1.0, 'mode': 'fan_avg', 'distribution': 'uniform', 'seed': None}}, 'bias_initializer': {'class_name': 'Zeros', 'config': {}}, 'kernel_regularizer': None, 'bias_regularizer': None, 'activity_regularizer': None, 'kernel_constraint': None, 'bias_constraint': None}
 *
**/


int main(int argc, char* argv[]) {


	cout << getCurrentWorkingDir() << endl;

	/// load the data set
	mnist::MNIST_dataset<uint8_t, uint8_t> dataset = loadMNIST();

	/// create a ciphertext factory. The layers need to know how to create new ciphertexts
	HELibCipherTextFactory ctxtFactory( 0, false );


	/// create Tensor factories for both weights and data
	HETensorFactory<HELibCipherText> hetfactory( &ctxtFactory );
	PlainTensorFactory<long> ptFactory;

	/// the bachsize is dependent on the parameters of the HE scheme
	uint batchSize = ctxtFactory.batchsize();

	/// We need to define an input tensor for the Model. This tensor needs to do know the shape of the data.
	/// The shape for all consecutive layers will be calculated as they are added to the model.
	///	input shape should be batch, channels, height, width. Since a single cipher text contains a number of
	/// plaintexts the batchsize is set to 1.
	TensorP<HELibCipherText> input = hetfactory.create( Shape( { 1, 1, 28, 28 } ) );

	/// instantiate model
	/// we need to specify the way memory is allocated and pass factories for the data tensors and weight tensors
	Model<HELibCipherText, long, HETensor<HELibCipherText>, PlainTensor<long> > completeNetwork( MemoryUsage::greedy,
			&hetfactory,
			&ptFactory );

	/// add layers
	/// the first layer needs to be passed an input tensor. all other tensors will be created automatically
	//conv1
	completeNetwork.addLayer(
			std::make_shared<Convolution2D<HELibCipherText, long, HETensor<HELibCipherText>, PlainTensor<long> >>(
					"conv2d_1", SquareActivation<HELibCipherText>::getSharedPointer(), 32, 5, 2, PADDING_MODE::SAME,
					input, &hetfactory,
					&ptFactory ) );
	//conv2
	completeNetwork.addLayer(
			std::make_shared<Convolution2D<HELibCipherText, long, HETensor<HELibCipherText>, PlainTensor<long> >>(
					"conv2d_2", LinearActivation<HELibCipherText>::getSharedPointer(), 64, 5, 2, PADDING_MODE::VALID,
					&hetfactory,
					&ptFactory ) );

	// flatten
	completeNetwork.addLayer(
			std::make_shared<Flatten<HELibCipherText, long, HETensor<HELibCipherText>, PlainTensor<long> >>( "flatten",
					&hetfactory, &ptFactory ) );
	//dense 1
	completeNetwork.addLayer(
			std::make_shared<Dense<HELibCipherText, long, HETensor<HELibCipherText>, PlainTensor<long> >>( "dense_1",
					SquareActivation<HELibCipherText>::getSharedPointer(), 100, &hetfactory, &ptFactory ) );

	//dense 2
	completeNetwork.addLayer(
			std::make_shared<Dense<HELibCipherText, long, HETensor<HELibCipherText>, PlainTensor<long> >>( "dense_2",
					LinearActivation<HELibCipherText>::getSharedPointer(), 10, &hetfactory, &ptFactory ) );

	/// since we have named the layers the same way they were named in keras we only need to specify the directory that
	/// the exported weights
	completeNetwork.loadWeights( "examples/mnist_cnn/weights/" );
	std::cout << "batchsize: " << batchSize << std::endl;

	/// crop the testdata to fit our batchsize
	std::vector<std::vector<uint8_t>> X( dataset.test_images.begin(), dataset.test_images.begin() + batchSize );
	std::vector<uint8_t> Y( dataset.test_labels.begin(), dataset.test_labels.begin() + batchSize );

	/// encrypt the test data
	/// The "shape" of the data as it is loaded is [ instances, pixels ] but we need to reshape it to [ pixels, instances ]
	/// so that we can encrypt all pixels with the same coordinates from multiple images into one ciphertext.
	std::vector<std::vector<HELibCipherText>> encImages( 1, std::vector<HELibCipherText>() ); /// nested vectors to simulate batches
	for ( size_t i = 0; i < X [ 0 ].size(); i++ ) {
		/// contains every i th from all images in the test test batch
		vector<long> lp;
		/// iterate over all images in the batch and extract ith pixel
		for ( uint b = 0; b < batchSize; b++ ) {
			lp.push_back( unsigned( dataset.test_images [ b ] [ i ] ) );
		}
		/// create a ciphertext contatining all the ith pixles
		auto enc = ctxtFactory.createCipherText( lp );
		encImages [ 0 ].push_back( enc );
	}


	try {
		completeNetwork.evaluate<HELibCipherText, uint8_t>( encImages, Y ); /// evaluate throws an excpetion cause we can not
																			/// not do predictions on encrypted data
	} catch ( ... ) {
		// ignore the exception
	}

	/// get teh output of the network and decrypt it
	TensorP<double> decrypted = ( (HETensor<HELibCipherText>*) completeNetwork.mLayers.back()->output().get() )->decryptDouble();
	/// calcuate the accuracy and print results
	auto preds = decrypted->argmaxVector();
	float acc = completeNetwork.accuracy( preds, Y );
	std::cout << preds << std::endl;
	std::cout << "[";
	for ( auto p : Y )
		std::cout << (unsigned) p << " ";
	std::cout << "]" << std::endl;
	cout << "Accuracy: " << acc << std::endl;







	return 0;

}
