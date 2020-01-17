# JSON Model Loading 

## Backend

This project uses [nlohmann's json parser](https://github.com/nlohmann/json) to load
json files and create models from them using our architecture.

## Custom Activation Functions
To use custom activation functions, extend the Activation class in your Keras model
through Python. e.g.:
```
# extend activation class
class Square(Activation):
    def __init__(self, activation, **kwargs):
        super(Square, self).__init__(activation, **kwargs)
        self.__name__ = 'square'

# activation function
def square(x):
	return K.square(x)

# update custom objects
keras.utils.get_custom_objects().update( { 'square': Square(square) } )
```
Then, you will need to add the appropriate name in JSONModel.h's `grabActivation()`
and function in `architecture/ActivationFunctionImpl.h`. 

## Memory Usage
When creating a model using JSONModel.h's `fromFile()`, you can dictate what type of
memory usage you would like. Although we currently only allow for greedy usage (and it
acts as a default param for creation), there are plans to allow for a lazy memory usage
instantiation.

## Layer Attributes
As we currently only allow for Conv2D, Dense, and Flatten layer functionality, these are
the attributes that we will focus on.

### Conv2D
The Keras json file provides us with the following attributes:
`kernel_initializer, bias_regularizer, dilation_rate, activity_regularizer, activation, data_format, filters, kernel_regularizer, use_bias, kernel_constraint, name, bias_initializer, kernel_size, dtype, trainable, batch_input_shape, bias_constraint, strides, padding`

We only use the following attributes in our architecture:
`name, activation, filters, kernel_size, strides, padding` and sometimes `batch_input_shape` (when it is the first layer in the model)

### Dense
The Keras json file provides us with the following attributes:
`bias_regularizer, units, activation, bias_initializer, dtype, kernel_regularizer, use_bias, kernel_constraint, kernel_initializer, name, trainable, bias_constraint, activity_regularizer, batch_input_shape`

We only use the following attributes in our architecture:
`name, activation,  units` and sometimes `batch_input_shape` (when it is the first layer in the model)


### Flatten
The Keras json file provides us with the following attributes:
`data_format, trainable, name`

We only use the following attributes in our architecture:
`name`