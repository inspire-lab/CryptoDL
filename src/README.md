# src Files

## CNNENC.cpp
```
The main program. When passed-in with specific arguments,the logic of the program will change.
The arguments are as follow:
{"Mode: "{tests, experiments}}
If the only argument passed in is 'tests', then the program will run the unit tests in tests/Tests.h.  
If the only argument passed in is 'experiments', then the program will create a HElib-backend model  
and run the classification of the encrypted MNIST dataset. 
```

## DatasetOperations.*
```
The files which handle the loading of the MNIST dataset through the use of templating.
(Special thanks to:) https://github.com/wichtounet/mnist
```
