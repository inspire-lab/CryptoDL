/*
 * PoolingTest.h
 *
 *  Created on: Apr 3, 2019
 *      Author: agustin
 */

#ifndef TEST_POOLINGTEST_H_
#define TEST_POOLINGTEST_H_

#include <gmpxx.h>
#include <errno.h>
#include "TestCommons.h"
#include "../src/architecture/ActivationFunction.h"
#include "../src/architecture/Layer.h"
#include "../src/architecture/PlainTensor.h"
#include "../src/architecture/Model.h"

using namespace std;

bool evenPoolValidOnesTest1();
bool evenPoolValidOnesTest2();
bool evenPoolValidOnesTest3();
bool oddPoolValidOnesTest1();
bool oddPoolValidOnesTest2();

bool evenPoolValidRangeTest1();
bool evenPoolValidRangeTest2();
bool oddPoolValidRangeTest1();

bool evenPoolSameOnesTest1();
bool evenPoolSameOnesTest2();

bool evenPoolSameRangeTest1();
bool evenPoolSameRangeTest2();
bool evenPoolSameRangeTest3();

bool oddPoolSameOnesTest1();
bool oddPoolSameOnesTest2();

bool oddPoolSameRangeTest1();
bool oddPoolSameRangeTest2();
bool oddPoolSameRangeTest3();


#endif /* TEST_POOLINGTEST_H_ */
