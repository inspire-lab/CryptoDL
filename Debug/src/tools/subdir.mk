################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/tools/Config.cpp \
../src/tools/DataReaders.cpp \
../src/tools/FileSystemTools.cpp \
../src/tools/RNNTools.cpp \
../src/tools/SystemTools.cpp 

OBJS += \
./src/tools/Config.o \
./src/tools/DataReaders.o \
./src/tools/FileSystemTools.o \
./src/tools/RNNTools.o \
./src/tools/SystemTools.o 

CPP_DEPS += \
./src/tools/Config.d \
./src/tools/DataReaders.d \
./src/tools/FileSystemTools.d \
./src/tools/RNNTools.d \
./src/tools/SystemTools.d 


# Each subdirectory must supply rules for building sources it contributes
src/tools/%.o: ../src/tools/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++17 -O3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


