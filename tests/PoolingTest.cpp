
/*
 * PoolingTest.cpp
 *
 *  Created on: Apr 3, 2019
 *      Author: agustin
 */
#include "PoolingTest.h"

bool evenPoolValidOnesTest1() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.onesAndInit( { 1, 1, 9, 9 } );

	//setup expected output
	vector<vector<vector<vector<float>>>> expectedOutputV = vector<
			vector<vector<vector<float>>>> { vector<vector<vector<float>>> {
			vector<vector<float>>( 8, vector<float>( 8, 1 ) ) } }; //expected output is 8x8 all ones
	TensorP<float> expectedOutput = dataFactory.create(
			Shape( { 1, 1, 8, 8 } ) );
	expectedOutput->init( expectedOutputV );

	//setup layer and run
	int filterSize = 2, stride = 1;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::VALID, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

bool evenPoolValidOnesTest2() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.onesAndInit( { 1, 1, 9, 9 } );

	//setup expected output
	vector<vector<vector<vector<float>>>> expectedOutputV = vector<
			vector<vector<vector<float>>>> { vector<vector<vector<float>>> {
			vector<vector<float>>( 4, vector<float>( 4, 1 ) ) } }; //expected output is 8x8 all ones
	TensorP<float> expectedOutput = dataFactory.create(
			Shape( { 1, 1, 4, 4 } ) );
	expectedOutput->init( expectedOutputV );

	//setup layer and run
	int filterSize = 2, stride = 2;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::VALID, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

bool evenPoolValidOnesTest3() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.onesAndInit( { 1, 1, 9, 9 } );

	//setup expected output
	vector<vector<vector<vector<float>>>> expectedOutputV = vector<
			vector<vector<vector<float>>>> { vector<vector<vector<float>>> {
			vector<vector<float>>( 3, vector<float>( 3, 1 ) ) } }; //expected output is 8x8 all ones
	TensorP<float> expectedOutput = dataFactory.create(
			Shape( { 1, 1, 3, 3 } ) );
	expectedOutput->init( expectedOutputV );

	//setup layer and run
	int filterSize = 4, stride = 2;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::VALID, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

bool oddPoolValidOnesTest1() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.onesAndInit( { 1, 1, 9, 9 } );

	//setup expected output
	vector<vector<vector<vector<float>>>> expectedOutputV = vector<
			vector<vector<vector<float>>>> { vector<vector<vector<float>>> {
			vector<vector<float>>( 4, vector<float>( 4, 1 ) ) } }; //expected output is 8x8 all ones
	TensorP<float> expectedOutput = dataFactory.create(
			Shape( { 1, 1, 4, 4 } ) );
	expectedOutput->init( expectedOutputV );

	//setup layer and run
	int filterSize = 3, stride = 2;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::VALID, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

bool oddPoolValidOnesTest2() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.onesAndInit( { 1, 1, 10, 10 } );

	//setup expected output
	vector<vector<vector<vector<float>>>> expectedOutputV = vector<
			vector<vector<vector<float>>>> { vector<vector<vector<float>>> {
			vector<vector<float>>( 3, vector<float>( 3, 1 ) ) } }; //expected output is 8x8 all ones
	TensorP<float> expectedOutput = dataFactory.create(
			Shape( { 1, 1, 3, 3 } ) );
	expectedOutput->init( expectedOutputV );

	//setup layer and run
	int filterSize = 5, stride = 2;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::VALID, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

bool evenPoolValidRangeTest1() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.arangeAndInit( { 1, 1, 6, 6 } );

	//setup expected output
//Output:
//[[[[  4.5   6.5   8.5]
//   [ 16.5  18.5  20.5]
//   [ 28.5  30.5  32.5]]]]
	vector<vector<vector<vector<float>>>> expectedOutputV { vector<
			vector<vector<float>>> { vector<vector<float>> { vector<float> {
			4.5, 6.5, 8.5 }, vector<float> { 16.5, 18.5, 20.5 }, vector<float> {
			28.5, 30.5, 32.5 } } } };
	TensorP<float> expectedOutput = dataFactory.create(
			Shape( { 1, 1, 3, 3 } ) );
	expectedOutput->init( expectedOutputV );

	//setup layer and run
	int filterSize = 2, stride = 2;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::VALID, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

bool evenPoolValidRangeTest2() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.arangeAndInit( { 1, 1, 8, 8 } );

	//setup expected output
	//TODO: fix having to remove spaces
	std::string kerasOutput =
			"[[[[  5.5   6.5   7.5   8.5   9.5  10.5  11.5][ 13.5  14.5  15.5  16.5  17.5  18.5  19.5][ 21.5  22.5  23.5  24.5  25.5  26.5  27.5][ 29.5  30.5  31.5  32.5  33.5  34.5  35.5][ 37.5  38.5  39.5  40.5  41.5  42.5  43.5][ 45.5  46.5  47.5  48.5  49.5  50.5  51.5][ 53.5  54.5  55.5  56.5  57.5  58.5  59.5]]]]";

	TensorP<float> expectedOutput = createTensorFromKeras<float>(
			{ 1, 1, 7, 7 }, kerasOutput );

	//setup layer and run
	int filterSize = 2, stride = 1;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::VALID, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

bool oddPoolValidRangeTest1() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.arangeAndInit( { 1, 1, 10, 10 } );

	//setup expected output
	//TODO: fix having to remove spaces
	std::string kerasOutput =
			"[[[[ 23.  25.  27.][ 43.  45.  47.][ 63.  65.  67.]]]]]";

	TensorP<float> expectedOutput = createTensorFromKeras<float>(
			{ 1, 1, 3, 3 }, kerasOutput );

	//setup layer and run
	int filterSize = 5, stride = 2;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::VALID, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

///move arange and allones tensors to abstract tensor class as pure virtual
//implement helib version

bool oddPoolValidRangeTest2() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.arangeAndInit( { 1, 1, 10, 10 } );

	//setup expected output
	//TODO: fix having to remove spaces
	std::string kerasOutput =
			"[[[[ 23.  25.  27.][ 43.  45.  47.][ 63.  65.  67.]]]]]";

	TensorP<float> expectedOutput = createTensorFromKeras<float>(
			{ 1, 1, 3, 3 }, kerasOutput );

	//setup layer and run
	int filterSize = 3, stride = 3;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::VALID, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

bool evenPoolSameOnesTest1() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.onesAndInit( { 1, 1, 9, 9 } );

	//setup expected output
	vector<vector<vector<vector<float>>>> expectedOutputV = vector<
			vector<vector<vector<float>>>> { vector<vector<vector<float>>> {
			vector<vector<float>>( 9, vector<float>( 9, 1 ) ) } }; //expected output is 8x8 all ones
	TensorP<float> expectedOutput = dataFactory.create(
			Shape( { 1, 1, 9, 9 } ) );
	expectedOutput->init( expectedOutputV );

	//setup layer and run
	int filterSize = 2, stride = 1;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::SAME, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

bool evenPoolSameOnesTest2() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.onesAndInit( { 1, 1, 6, 6 } );

	//setup expected output
	TensorP<float> expectedOutput = dataFactory.onesAndInit( { 1, 1, 3, 3 } );

	//setup layer and run
	int filterSize = 4, stride = 2;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::SAME, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

bool evenPoolSameRangeTest1() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.arangeAndInit( { 1, 1, 10, 10 } );

	//setup expected output
	std::string kerasOutput =
			"[[[[ 6.5 7.5 8.5 9.5 10.5 11.5 12.5 13.5 14.5 15. ][ 16.5 17.5 18.5 19.5 20.5 21.5 22.5 23.5 24.5 25. ][ 26.5 27.5 28.5 29.5 30.5 31.5 32.5 33.5 34.5 35. ][ 36.5 37.5 38.5 39.5 40.5 41.5 42.5 43.5 44.5 45. ][ 46.5 47.5 48.5 49.5 50.5 51.5 52.5 53.5 54.5 55. ][ 56.5 57.5 58.5 59.5 60.5 61.5 62.5 63.5 64.5 65. ][ 66.5 67.5 68.5 69.5 70.5 71.5 72.5 73.5 74.5 75. ][ 76.5 77.5 78.5 79.5 80.5 81.5 82.5 83.5 84.5 85. ][ 86.5 87.5 88.5 89.5 90.5 91.5 92.5 93.5 94.5 95. ][ 91.5 92.5 93.5 94.5 95.5 96.5 97.5 98.5 99.5 100. ]]]]";
	TensorP<float> expectedOutput = createTensorFromKeras<float>(
			{ 1, 1, 10, 10 }, kerasOutput );

	//setup layer and run
	int filterSize = 2, stride = 1;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::SAME, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

bool evenPoolSameRangeTest2() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.arangeAndInit( { 1, 1, 5, 5 } );

	//setup expected output
	std::string kerasOutput =
			"[[[[  4.    5.    6.    7.    7.5][  9.   10.   11.   12.   12.5][ 14.   15.   16.   17.   17.5][ 19.   20.   21.   22.   22.5][ 21.5  22.5  23.5  24.5  25. ]]]]";
	TensorP<float> expectedOutput = createTensorFromKeras<float>(
			{ 1, 1, 5, 5 }, kerasOutput );

	//setup layer and run
	int filterSize = 2, stride = 1;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::SAME, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

bool evenPoolSameRangeTest3() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.arangeAndInit( { 1, 1, 6, 6 } );

	//setup expected output
	std::string kerasOutput =
			"[[[[  4.5   6.5   8.5][ 16.5  18.5  20.5][ 28.5  30.5  32.5]]]]";
	TensorP<float> expectedOutput = createTensorFromKeras<float>(
			{ 1, 1, 3, 3 }, kerasOutput );

	//setup layer and run
	int filterSize = 2, stride = 2;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::SAME, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

bool oddPoolSameOnesTest1() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.onesAndInit( { 1, 1, 6, 6 } );

	//setup expected output
	TensorP<float> expectedOutput = dataFactory.onesAndInit( { 1, 1, 3, 3 } );

	//setup layer and run
	int filterSize = 3, stride = 2;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::SAME, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

bool oddPoolSameOnesTest2() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.onesAndInit( { 1, 1, 15, 15 } );

	//setup expected output
	TensorP<float> expectedOutput = dataFactory.onesAndInit( { 1, 1, 5, 5 } );

	//setup layer and run
	int filterSize = 5, stride = 3;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::SAME, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

bool oddPoolSameRangeTest1() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.arangeAndInit( { 1, 1, 10, 10 } );

	//setup expected output
	std::string kerasOutput =
			"[[[[ 12.   16.   19.5][ 52.   56.   59.5][ 87.   91.   94.5]]]]";
	TensorP<float> expectedOutput = createTensorFromKeras<float>(
			{ 1, 1, 3, 3 }, kerasOutput );

	//setup layer and run
	int filterSize = 3, stride = 4;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::SAME, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

bool oddPoolSameRangeTest2() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.arangeAndInit( { 1, 1, 8, 8 } );

	//setup expected output
	std::string kerasOutput =
			"[[[[ 10.  14.][ 42.  46.]]]]";
	TensorP<float> expectedOutput = createTensorFromKeras<float>(
			{ 1, 1, 2, 2 }, kerasOutput );

	//setup layer and run
	int filterSize = 3, stride = 4;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::SAME, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}

bool oddPoolSameRangeTest3() {
	cout << "Running " << __func__ << " " << flush;

	//create necessary factories and inputs/weights
	PlainTensorFactory<float> dataFactory, weightFactory;

	//setup input
	TensorP<float> input = dataFactory.arangeAndInit( { 1, 1, 24, 24 } );

	//setup expected output
	std::string kerasOutput =
			"[[[[  38.5   40.    42.    44.    46.    48.    50.    52.    54.    56.    58.    59.	][ 74.5 76. 78. 80. 82. 84. 86. 88. 90. 92.	94. 95. ][ 122.5 124. 126. 128. 130. 132. 134. 136. 138. 140. 142. 143. ][ 170.5 172. 174. 176. 178. 180. 182. 184. 186. 188. 190. 191. ][ 218.5 220. 222. 224. 226. 228. 230. 232. 234. 236. 238. 239. ][ 266.5 268. 270. 272. 274. 276. 278. 280. 282. 284. 286. 287. ][ 314.5 316. 318. 320. 322. 324. 326. 328. 330. 332. 334. 335. ][ 362.5 364. 366. 368. 370. 372. 374. 376. 378. 380. 382. 383. ][ 410.5 412. 414. 416. 418. 420. 422. 424. 426. 428. 430. 431. ][ 458.5 460. 462. 464. 466. 468. 470. 472. 474. 476. 478. 479. ][ 506.5 508. 510. 512. 514. 516. 518. 520. 522. 524. 526. 527. ][ 530.5 532. 534. 536. 538. 540. 542. 544. 546. 548. 550. 551. ]]]]";
	TensorP<float> expectedOutput = createTensorFromKeras<float>(
			{ 1, 1, 12, 12 }, kerasOutput );

	//setup layer and run
	int filterSize = 5, stride = 2;
	AveragePooling<float, float, float, float> pool( "test",
			LinearActivation<float>::getSharedPointer(), filterSize, stride,
			PADDING_MODE::SAME, input, &dataFactory, &weightFactory );
	pool.output()->init();
	pool.feedForward();

	//compare output w/ expected output
	return finishTest( pool.output(), expectedOutput, __func__ );
}
