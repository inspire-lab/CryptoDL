/*
 * HEBackendTests.h
 *
 *  Created on: Mar 3, 2019
 *      Author: robert
 */

#ifndef TEST_HEBACKENDTESTS_H_
#define TEST_HEBACKENDTESTS_H_


bool HE_convTest1_samePad();

bool HE_convTest2_samePad();

bool HE_convTest1_validPad();

bool HE_convTest2_validPad();

bool HE_denseTest1();

bool HE_denseTest2();

bool HE_convTest1_samePad_CKKS();

bool HE_convTest1_samePadBatch_CKKS();

bool HE_SamePaddFloats();

bool HE_convTest1_ValidBatch_CKKS();

bool HE_convTest2_Valid_CKKS();

#endif /* TEST_HEBACKENDTESTS_H_ */
