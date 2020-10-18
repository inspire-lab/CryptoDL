################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/test/CompleteNetworkTests.cpp \
../src/test/ConvTestSamePadding.cpp \
../src/test/ConvTestValidPadding.cpp \
../src/test/DenseTest.cpp \
../src/test/HEBackendTests.cpp \
../src/test/PoolingTest.cpp \
../src/test/RNNTest.cpp 

OBJS += \
./src/test/CompleteNetworkTests.o \
./src/test/ConvTestSamePadding.o \
./src/test/ConvTestValidPadding.o \
./src/test/DenseTest.o \
./src/test/HEBackendTests.o \
./src/test/PoolingTest.o \
./src/test/RNNTest.o 

CPP_DEPS += \
./src/test/CompleteNetworkTests.d \
./src/test/ConvTestSamePadding.d \
./src/test/ConvTestValidPadding.d \
./src/test/DenseTest.d \
./src/test/HEBackendTests.d \
./src/test/PoolingTest.d \
./src/test/RNNTest.d 


# Each subdirectory must supply rules for building sources it contributes
src/test/%.o: ../src/test/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ $(CXXFLAGS) $(INCLUDES) -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


