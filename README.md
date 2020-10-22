# THIS IS OUTDATED! WILL BE UPDATED SOON!


# CryptoDL

# README 
v.0.1.0

## What is this repository for? ##

This project aims to provide Deep Learning with a privacy preserving computation
backend. The privacy preserving computation is based on homomorphic encryption
(HE). Homomorphic encryption allows for computation on encrypted data, get an
encrypted result without the need to decrypt the data for any of the computation.
These properties come with a number of restrictions and things to look out for.
We aim to provide a library that provides privacy preserving deep learning for
people with a deep learning background without needing to dig into HE, aswell as
the abiltity to easily integrate different HE libraries as computation backends. 


## Things to be aware of and look out for ##

### Image data format and channels  ###

Out system currently assumes that the image data format is **channels first**.
This means the contains the inputs are of the following form 
`[batch, channel, y, x]`. 

#### Keras 2.0.6 and later ####

In Keras 2.0.6 the `Flatten` layer became aware of the image data format. In
prior versions it was always assuming channels last. Our implementation does 
not suport channels first flattening in the same way that Keras does. So make
sure that your `Flatten` layers contains the correct data format like this:

```
Flatten(data_format="channels_last")
```
 
#### Supported version of HELib ####

The last known version of HELib supported is: `ac0308715e5ae6bf5e750e8701e736d855550fc8`
To obtain it use:

```
git checkout ac0308715e5ae6bf5e750e8701e736d855550fc8
```
#### HELib build issues #### 

There are issues witht he new HELib build system. It produces a way slower library.
See the build section how to use the old system.

 
## What is supported at the moment? ##




### Deep Learning

So far we support the following types of layers:

- 2D Convolutional
- Fully Connected
- Fully Connected Recurrent Layer

We support a number of activation functions:
- Linear
- Square
- ReLU ( not useable with the HE backend )
- Polynomials 

We **only** support inference. Training is not supported and needs to be done
with other tools/frameworks. As of now there is no automated way of importing 
pretrained models. The current suggested workflow is: 
1. Train model using [keras](https://keras.io/)
2. Use provided tools to extract weights
3. Define model in C++ and import weights
Models that are supposed to used with our code may only use these layers and
activation functions.
  

### HE Backend

Currently we only support [HELib](https://github.com/shaih/HElib) library as a 
computational backend (other than plaintext).

#### Limitations

HE places a number of constraints and limitations on the type of computation
that can be performed. This is supposed to be a list of the main points to 
be aware off. It is not meant to be comprehenisve.

- **No Division** Division of ciphertexts is not supported
- **No comparison** There is no comparison between ciphertext, e.g. no max,
min, etc. This means we can not use ReLU, MaxPooling, etc. 
- **Limited number of computations** Every computation performed on a 
ciphertext adds some additional noise to the ciphertext. If the noise 
exceeds some threshold we can not decrypt it correctly anymore. Therefore we
can not run an arbitrary number of computations
- **Limited support of Activation functions, Layers** Due the constraints 
mentioned above we can not use every activation function or layer that we 
normally can. We suggest using poynomials as activation functions. 

## How do I get set up? ##

### Dependencies ##



C++:
 - HELib
 - boost 1.67
 - libjpeg-dev

python:
 - keras
 - tenserflow


At the moment only Linux is supported. We have run it succesfully on Ubuntu 16.04,  
Ubuntu 17.10, Ubuntu 18.10 and Ubunutu 19.04. Other versions might work as well. 
Our project is based on [HELib](https://github.com/shaih/HElib) which needs to be
built. 

### Building HElib

The build instructions have changed. There are performance issue with the libary 
file that is produced by the new HElib build system. Use the legacy system for
now. 

1. Follow the instructions here: https://github.com/shaih/HElib/blob/master/OLD_INSTALL.txt
When  installling `gmp` make sure to include the c++ interface by using ` ./configure --enable-cxx `
2. We expect to find the HElib headers in the systems include path. Typically 
 `/usr/local/include`. To make sure they can be found there create a directory called
`helib` in `/usr/local/include` and copy all the `.h` files from the `HElib/src` 
directory there.
3. After installing HELib go into the `build` directory
4. In the `objects.mk` file make sure the path to `fhe.a` is correct for your system
5. Call `make`
6. The resulting binary `CNNENC` needs to be copied to the project to be run correctly

~~1. Install [HELib](https://github.com/shaih/HElib) and its dependencies. Follow the instrutctions for Option 2 library installation.~~

### Building CryptoDL

1. Check out this project.
2. Build the library using the makefile located in `Debug`
3. The output is `Debug/libkalypso.a`

If the project build fails on the HELib include, copy the `.h` files from the HElib directory to your global include directory.

### Installing the python module

Some parts of the project require our python module. To install it run

`pip install -e python`



### What next?

The `libkalypso.a` library can be used to build deeplearning models using the HEbackend. Be sure to add the `src` folder to the include path.

For tips to how get started. Checkout the `examples` directory.


## Contribution guidelines ##

TBD

## Who do I talk to? ##
Contact: rpodschwadt1@student.gsu.edu
