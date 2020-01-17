################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/experiments/alexnet/AlexNet.cpp \
../src/experiments/alexnet/AlexNet32.cpp 

OBJS += \
./src/experiments/alexnet/AlexNet.o \
./src/experiments/alexnet/AlexNet32.o 

CPP_DEPS += \
./src/experiments/alexnet/AlexNet.d \
./src/experiments/alexnet/AlexNet32.d 


# Each subdirectory must supply rules for building sources it contributes
src/experiments/alexnet/%.o: ../src/experiments/alexnet/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -std=c++0x -include../src/mnist/include/mnist/mnist_reader_less.hpp -O3 -g0 -Wall -c -fmessage-length=0 -fopenmp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


