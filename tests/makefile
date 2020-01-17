# All Target

SRC_FILES := $(wildcard *.cpp)
OBJ_FILES := $(patsubst %.cpp,%.o,$(SRC_FILES))
DEP_FILES := $(patsubst %.cpp,%.d,$(SRC_FILES))


all: $(OBJ_FILES)
	@echo Linking tests
	g++ -o kalypso_tests $(OBJ_FILES) ../../HElib-master/src/fhe.a ../Debug/libkalypso.a  -lntl -lboost_filesystem -lboost_system -ljpeg -lpthread -lgmpxx -lgmp  

%.o: %.cpp
	@echo Building tests
	g++ -std=c++0x -include../src/data/mnist/include/mnist/mnist_reader_less.hpp -O0 -g3 -Wall -c -fmessage-length=0 -fopenmp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o $@ $<

clean:
	rm kalypso_tests $(OBJ_FILES) $(DEP_FILES) 
	
