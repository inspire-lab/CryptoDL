# CryptoDL

# How to install

These steps assume your are running Ubuntu.

## First step

Clone this repository
 
## Install Dependency Packages

`CryptoDL/dependencies/` contains scripts and information to install and build the 
required dependencies. 

Run
```config_system.sh``` 
as root to install the depencies that are availabe via the package manager.

After that run
```
install_dependencies.sh
```
from `CryptoDL/dependencies/` directory. This downloads and builds dependcies
not only availalbe from source or that require specific configuration.

## Building CryptoDL

To build CrptoDL simply run 
```
make
```
in the `CryptoDL/Debug` directory. This builds a static library `libkalypso.a` which
other projects can link against.


To make linking easier `CryptoDL/dependencies/versions.sh` creates varialbes that 
you can use in your build scripts.

An example build script can looks like this. All you need to is set the `CRYPTODL_DIR`
variable to where you coloned the repository.

```
$(shell $(CRYPTODL_DIR)/dependencies/versions.sh)
-include $(CRYPTODL_DIR)/dependencies/makefile.versions

g++ -std=c++17 -Wall $(INCLUDE_DIRS) $(DEP_INCLUDES) -c example.cpp
g++ -std=c++17 -o "example" example.o $(CRYPTODL_LIB) $(DEP_LIBS) $(DEP_RPATH)

```

## What is this repository for? ##

This project aims to provide Deep Learning with a privacy preserving computation
backend. The privacy preserving computation is based on homomorphic encryption
(HE). Homomorphic encryption allows for computation on encrypted data, get an
encrypted result without the need to decrypt the data for any of the computation.
These properties come with a number of restrictions and things to look out for.
We aim to provide a library that provides privacy preserving deep learning for
people with a deep learning background without needing to dig into HE, aswell as
the abiltity to easily integrate different HE libraries as computation backends. 


 
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


## Contribution guidelines ##

TBD

## Who do I talk to? ##
Contact: rpodschwadt1@student.gsu.edu
