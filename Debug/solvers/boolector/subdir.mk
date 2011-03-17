################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../solvers/boolector/boolector_conv.o \
../solvers/boolector/boolector_dec.o \
../solvers/boolector/boolector_get.o \
../solvers/boolector/boolector_prop.o 

CPP_SRCS += \
../solvers/boolector/boolector_conv.cpp \
../solvers/boolector/boolector_dec.cpp \
../solvers/boolector/boolector_get.cpp \
../solvers/boolector/boolector_prop.cpp 

OBJS += \
./solvers/boolector/boolector_conv.o \
./solvers/boolector/boolector_dec.o \
./solvers/boolector/boolector_get.o \
./solvers/boolector/boolector_prop.o 

CPP_DEPS += \
./solvers/boolector/boolector_conv.d \
./solvers/boolector/boolector_dec.d \
./solvers/boolector/boolector_get.d \
./solvers/boolector/boolector_prop.d 


# Each subdirectory must supply rules for building sources it contributes
solvers/boolector/%.o: ../solvers/boolector/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


