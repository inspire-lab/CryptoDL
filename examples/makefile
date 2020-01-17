# All Target

all: cnn rnn

cnn:
	@echo Building cnn examples
	g++ -std=c++0x -include../src/data/mnist/include/mnist/mnist_reader_less.hpp -O0 -g3 -Wall -c mnist_cnn/mnist_cnn.cpp
	@echo Linking rnn examples
	g++ -o mnist_cnn_example mnist_cnn.o ../Debug/libkalypso.a -lpthread  ../../HElib-master/src/fhe.a -lboost_filesystem -lboost_system -ljpeg -lgmpxx -lntl -lgmp
	
rnn:
	@echo Building rnn examples
	g++ -std=c++0x -include../src/data/mnist/include/mnist/mnist_reader_less.hpp -O0 -g3 -Wall -c mnist_rnn/mnist_rnn.cpp
	@echo Linking rnn examples
	g++ -o mnist_rnn_example mnist_rnn.o ../Debug/libkalypso.a -lpthread  ../../HElib-master/src/fhe.a -lboost_filesystem -lboost_system -ljpeg -lgmpxx -lntl -lgmp

clean:
	-rm mnist_cnn_example mnist_cnn.o mnist_rnn_example mnist_rnn.o
	
