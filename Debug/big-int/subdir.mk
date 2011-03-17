################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../big-int/bigint-func.o \
../big-int/bigint-test.o \
../big-int/bigint.o 

CC_SRCS += \
../big-int/bigint-func.cc \
../big-int/bigint-test.cc \
../big-int/bigint.cc 

OBJS += \
./big-int/bigint-func.o \
./big-int/bigint-test.o \
./big-int/bigint.o 

CC_DEPS += \
./big-int/bigint-func.d \
./big-int/bigint-test.d \
./big-int/bigint.d 


# Each subdirectory must supply rules for building sources it contributes
big-int/%.o: ../big-int/%.cc
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


