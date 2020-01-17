# test Files

## CompleteNetworkTests.h*
```
Files that handle the execution of a complete plain network test through various types, including: mpz_class, long, and float. Some of the functions also use 'all-ones' for the filters and biases.
```

## ConvTest.h
```
A file that prototypes convolutional testing functions on same padding, valid padding, and other attributes.
```

## ConvTestSamePadding.cpp
```
A file that defines the prototypes for the same-padding test functions in ConvTest.h
```

## ConvTestValidPadding.cpp
```
A file that defines the prototypes for the valid-padding test functions in ConvTest.h
Also defines a convolutional layer 2 test where the second conv. layer is tested through the use of Keras' actual output. 
```

## DenseTest.*
```
Files that contain the functions used to run tests on the flattenImages function which takes a templated 2-d vector and compares each element.
```

## PlainNetwork.h
```
A file that contains templated functions that compare datasets, load filters, compute the average on magnitude, and run a plain-image network from our model and prints the accuracy.
```

## TestCommons.h
```
The file that contains comparing, and finishTest functions used in the majority of test definitions.
compareOutput traverses two 3-dimensional vectors of longs, comparing each element and returns a bool that states whether test succeeded or not.
finishTest uses the compareOutput, and prints the results.
```

## TestCommonsImpl.h
```
File that contains templated functions for comparing output, finishing tests, and printing results.
```

## Tests.h
```
File that executes the tests defined in this folder. On the successful execution of all tests, it prints 'success', and prints an error otherwise. The function terminates the program regardless of the result.
```