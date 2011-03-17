################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../solvers/cvc/cvc_conv.o \
../solvers/cvc/cvc_dec.o \
../solvers/cvc/cvc_prop.o 

CPP_SRCS += \
../solvers/cvc/cvc_conv.cpp \
../solvers/cvc/cvc_dec.cpp \
../solvers/cvc/cvc_prop.cpp 

OBJS += \
./solvers/cvc/cvc_conv.o \
./solvers/cvc/cvc_dec.o \
./solvers/cvc/cvc_prop.o 

CPP_DEPS += \
./solvers/cvc/cvc_conv.d \
./solvers/cvc/cvc_dec.d \
./solvers/cvc/cvc_prop.d 


# Each subdirectory must supply rules for building sources it contributes
solvers/cvc/%.o: ../solvers/cvc/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


