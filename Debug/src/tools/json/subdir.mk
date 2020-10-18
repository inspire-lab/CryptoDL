################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/tools/json/JSONModel.cpp 

OBJS += \
./src/tools/json/JSONModel.o 

CPP_DEPS += \
./src/tools/json/JSONModel.d 


# Each subdirectory must supply rules for building sources it contributes
src/tools/json/%.o: ../src/tools/json/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


