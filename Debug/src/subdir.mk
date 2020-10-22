################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/CNNENC.cpp \
../src/DatasetOperations.cpp 

OBJS += \
./src/CNNENC.o \
./src/DatasetOperations.o 

CPP_DEPS += \
./src/CNNENC.d \
./src/DatasetOperations.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++  $(INCLUDES)  -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


