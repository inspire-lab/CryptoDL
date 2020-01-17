################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/architecture/HEBackend/helib/HELIbCipherText.cpp 

OBJS += \
./src/architecture/HEBackend/helib/HELIbCipherText.o 

CPP_DEPS += \
./src/architecture/HEBackend/helib/HELIbCipherText.d 


# Each subdirectory must supply rules for building sources it contributes
src/architecture/HEBackend/helib/%.o: ../src/architecture/HEBackend/helib/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -std=c++0x -include../src/mnist/include/mnist/mnist_reader_less.hpp -O0 -g3 -Wall -c -fmessage-length=0 -fopenmp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


