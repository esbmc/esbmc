################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../satabs/cegar_loop.cpp \
../satabs/cmdline_options.cpp \
../satabs/safety.cpp \
../satabs/satabs.cpp 

OBJS += \
./satabs/cegar_loop.o \
./satabs/cmdline_options.o \
./satabs/safety.o \
./satabs/satabs.o 

CPP_DEPS += \
./satabs/cegar_loop.d \
./satabs/cmdline_options.d \
./satabs/safety.d \
./satabs/satabs.d 


# Each subdirectory must supply rules for building sources it contributes
satabs/%.o: ../satabs/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


