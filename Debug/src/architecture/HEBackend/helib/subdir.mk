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
	@echo 'Invoking: GCC C++ Compiler'
	g++ -std=c++17 -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


