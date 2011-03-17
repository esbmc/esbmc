################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../intrep/intrep.o \
../intrep/irep_language.o 

CPP_SRCS += \
../intrep/irep_language.cpp 

OBJS += \
./intrep/irep_language.o 

CPP_DEPS += \
./intrep/irep_language.d 


# Each subdirectory must supply rules for building sources it contributes
intrep/%.o: ../intrep/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


