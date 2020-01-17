/*
 * Tests.cpp
 *
 *  Created on: Jan 24, 2019
 *      Author: robert
 */


#include "TestCommons.h"
#include "ConvTest.h"
#include "CompleteNetworkTests.h"
#include "DenseTest.h"
#include "HEBackendTests.h"
#include "RNNTest.h"
#include "PoolingTest.h"



int main(int argc, char **argv) {

	bool success = true;

	success &= convTest1_samePad();
	success &= convTest2_samePad();
	success &= convTest3_samePad();
	success &= convTest4_samePad();
	success &= convTest5_samePad();
	success &= convTest6_samePad();
	success &= convTest7_samePad();
	convTest_samePad_evenKernel1();
//	success &= convTest_multiple_filters_multiple_channels_same_padding_1();

	success &= convTest1_validPad();
	success &= convTest2_validPad();
	success &= convTest3_validPad();
	success &= convTest4_validPad();
	success &= convTest5_validPad();

	success &= convTest_strides_samePadding_1();
	success &= convTest_multiple_channels_non_unit_stride_same_padding_1();
	success &= convTest_multiple_filters__multiple_channels_non_unit_stride_same_padding_1();
	success &= convTest_strides_validPad_1(); //works
	success &= convTest_multiple_channels_non_unit_stride_validPad_1(); //CAUSES ERROR


	success &= completeNetworkTestLong();
	success &= completeNetworkTestHELibBFV();
	success &= completeNetworkTestHELibCKKS();
	success &= completeNetworkTestFloat();

	success &= compareLayerByLayerLong();




	success &= flattenTest1();
	success &= flattenTest2();
	success &= flattenTest3();
	success &=  flattenTest4();
	success &= denseTest1();
	success &= denseTest2();


	success &= convTest_validPad_secondLayer_cryptonet();

/////// HE Tests
	success &= HE_convTest1_samePad();
	success &= HE_denseTest1();
	success &= HE_denseTest2();
	success &= HE_convTest1_samePad();


///// RNN tests
	success &= rnnTest1();
	success &= rnnTest2();
	success &= rnnTest3();

	success &= HE_convTest1_samePad_CKKS();
	success &= HE_convTest1_samePadBatch_CKKS();

	success &= compareLayerByLayerEncryptedFloat();
	success &= HE_convTest1_ValidBatch_CKKS();
	success &= HE_convTest2_Valid_CKKS();
	success &= compareLayerByLayerEncryptedFloat();
	success &= compareLayerByLayerEncryptedFloat();
	success &= HE_SamePaddFloats();
	success &= completeNetworkTestHELibCKKSFloatWeights();

	//pooling layer tests
	success &= evenPoolValidOnesTest1();
	success &= evenPoolValidOnesTest2();
	success &= evenPoolValidOnesTest3();
	success &= oddPoolValidOnesTest1();
	success &= oddPoolValidOnesTest2();
	success &= evenPoolValidRangeTest1();
	success &= evenPoolValidRangeTest2();
	success &= oddPoolValidRangeTest1();
	success &= evenPoolSameOnesTest1();
	success &= evenPoolSameOnesTest2();
	success &= evenPoolSameRangeTest1();
	success &= evenPoolSameRangeTest2();
	success &= evenPoolSameRangeTest3();
	success &= oddPoolSameOnesTest1();
	success &= oddPoolSameOnesTest2();
	success &= oddPoolSameRangeTest1();
	success &= oddPoolSameRangeTest2();
	success &= oddPoolSameRangeTest3();


	if(success){
		std::cout << "Test run successful" << std::endl;
		exit(0);
	}
	errno = -1;
	perror("Test run failed");
	exit(1);
}


