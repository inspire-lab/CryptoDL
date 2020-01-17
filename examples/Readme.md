#Readme

This directory contains two examples:
 - The mnist handwritten digit dataset with a CNN
 - The mnist handwritten digit dataset with an RNN
 
 
To compile the examples run `make all`. You need to build `libkalypso.a` first if you 
didn`t already. To run the the examples you need to download the dataset from here http://yann.lecun.com/exdb/mnist/ and unpack it.
In the `config.ini` you need to set the path to the files.

To run the examples you first need to extract the weights from the keras models. To do so run either `mnist_rnn.py` or `mnist_cnn.py` depending on which 
example you want to run. 

The compiled executable needs to be executed from the project root.