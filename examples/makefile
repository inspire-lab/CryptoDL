# All Target
CRYPTODL_DIR=..
$(shell $(CRYPTODL_DIR)/dependencies/versions.sh)
-include $(CRYPTODL_DIR)/dependencies/makefile.versions

all: cnn rnn

cnn:
	@echo Building cnn examples
	g++ -std=c++17 -Wall $(INCLUDE_DIRS) $(DEP_INCLUDES) -O0 -g3 -Wall -c mnist_cnn/mnist_cnn.cpp
	@echo Linking rnn examples
	g++ -std=c++17 -Wall -o mnist_cnn_example mnist_cnn.o $(CRYPTODL_DIR)/Debug/libkalypso.a $(DEP_LIBS) $(DEP_RPATH)
	
rnn:
	@echo Building rnn examples
	g++ -std=c++0x -include../src/data/mnist/include/mnist/mnist_reader_less.hpp -O0 -g3 -Wall -c mnist_rnn/mnist_rnn.cpp
	@echo Linking rnn examples
	g++ -o mnist_rnn_example mnist_rnn.o ../Debug/libkalypso.a -lpthread  ../../HElib-master/src/fhe.a -lboost_filesystem -lboost_system -ljpeg -lgmpxx -lntl -lgmp

clean:
	-rm mnist_cnn_example mnist_cnn.o mnist_rnn_example mnist_rnn.o
	
