################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/tools/Config.cpp \
../src/tools/DataReaders.cpp \
../src/tools/FileSystemTools.cpp \
../src/tools/SystemTools.cpp 

OBJS += \
./src/tools/Config.o \
./src/tools/DataReaders.o \
./src/tools/FileSystemTools.o \
./src/tools/SystemTools.o 

CPP_DEPS += \
./src/tools/Config.d \
./src/tools/DataReaders.d \
./src/tools/FileSystemTools.d \
./src/tools/SystemTools.d 


# Each subdirectory must supply rules for building sources it contributes
src/tools/%.o: ../src/tools/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -std=c++0x -include../src/data/mnist/include/mnist/mnist_reader_less.hpp -O3 -g0 -Wall -c -fmessage-length=0 -fopenmp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


