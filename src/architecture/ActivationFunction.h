/*
 * ActivationFunction.h
 *
 *  Created on: Feb 18, 2019
 *      Author: agustin
 */


#ifndef ACTIVATIONFUNCTION_H_
#define ACTIVATIONFUNCTION_H_

#include "ActivationFunctionImpl.h"


//i *was* thinking that maybe we have 1 class, with 1 activate function
//that takes the enum'd type to call the appropriate func
//so that extending this class consists of simply adding another enum and another function
//used to call upon the appropriate activation func w/ switch statement
//the argument against all that ^ would be that by having the abstract class, we only need to include
//the used activation functions
//enum Activation{ square, linear, relu, softmax };//don't think I will be using it




#endif /* ACTIVATIONFUNCTION_H_ */
