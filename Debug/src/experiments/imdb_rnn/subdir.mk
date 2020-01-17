################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/experiments/imdb_rnn/ImdbRNN.cpp 

OBJS += \
./src/experiments/imdb_rnn/ImdbRNN.o 

CPP_DEPS += \
./src/experiments/imdb_rnn/ImdbRNN.d 


# Each subdirectory must supply rules for building sources it contributes
src/experiments/imdb_rnn/%.o: ../src/experiments/imdb_rnn/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -std=c++0x -include../src/mnist/include/mnist/mnist_reader_less.hpp -O3 -g0 -Wall -c -fmessage-length=0 -fopenmp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


