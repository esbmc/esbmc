################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../satabs/prepare/concrete_model.cpp \
../satabs/prepare/map_vars.cpp \
../satabs/prepare/prepare.cpp 

OBJS += \
./satabs/prepare/concrete_model.o \
./satabs/prepare/map_vars.o \
./satabs/prepare/prepare.o 

CPP_DEPS += \
./satabs/prepare/concrete_model.d \
./satabs/prepare/map_vars.d \
./satabs/prepare/prepare.d 


# Each subdirectory must supply rules for building sources it contributes
satabs/prepare/%.o: ../satabs/prepare/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


