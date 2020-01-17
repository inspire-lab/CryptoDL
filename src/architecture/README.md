# The broad strokes

At the core it all is the `Model` class. A `Model` is a collection of `Layer` objects. A `Layer` provides the implementation
of deep learning layers. Every `Layer` has an input and output `Tensor`. The `Shape` of these `Tensors` gets computed dynamically
except for the Input `Tensor`. The implemenation of the `Layer` is based on the abstract `Tensor`. This allows for the HE 
functionallity to be encapsulated in the HEBackend. The HEBackend provides its own implementation of `Tensor` `HETensor`.
`HETensors` operate on `CipherTextWrapper` which encapsulate the actuall ciphertexts of the HE library and provide the arithmetic interface. The creation of specific `CipherTextWrappers` fall to the `CipherTextWrapperFactory`. This is the place where the encryption,
decryption and crypto parameters are being stored. For an example see `HEBackend/helib/HELibCiphertext.h`. 

`Tensor` and `CipherTextWrapper` objects are created by there respective factories. This is meant to hide the details of the crypto systems from higher leve code. The `CipherTextWrapperFactory` is encapsulated in the `HETensorFactory`. The high level deep learning code should work with abstract `Tensor` and `TensorFactory` objects only.  
      




# architecture Files

## activations/
This folder contains the implementations of activation functions the solution requires.

## ActivationFunction.h
Serves as an abstract class for any activation function the solution requires. Currently, the activations are  in
activations/ and consist of the Relu, Square, and Linear activation functions. 

## Layer.*
Contains the layer class which implements feed-forward on an input tensor, setting the activation, and printing a  
description of the layer. 

## Model.*
Contains the Model class which houses a container of layers, inputs/outputs, and generates the required memory.  
The class also handles printing of relevant model information.

## PlainTensor.h

## PlainTensorImpl.h

## Tensor.cpp

## Tensor.h

## TensorImpl.h
