################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../solvers/z3/z3_capi.o \
../solvers/z3/z3_conv.o \
../solvers/z3/z3_dec.o \
../solvers/z3/z3_get.o \
../solvers/z3/z3_prop.o \
../solvers/z3/z3_util.o 

CPP_SRCS += \
../solvers/z3/z3_capi.cpp \
../solvers/z3/z3_conv.cpp \
../solvers/z3/z3_dec.cpp \
../solvers/z3/z3_get.cpp \
../solvers/z3/z3_prop.cpp \
../solvers/z3/z3_util.cpp 

OBJS += \
./solvers/z3/z3_capi.o \
./solvers/z3/z3_conv.o \
./solvers/z3/z3_dec.o \
./solvers/z3/z3_get.o \
./solvers/z3/z3_prop.o \
./solvers/z3/z3_util.o 

CPP_DEPS += \
./solvers/z3/z3_capi.d \
./solvers/z3/z3_conv.d \
./solvers/z3/z3_dec.d \
./solvers/z3/z3_get.d \
./solvers/z3/z3_prop.d \
./solvers/z3/z3_util.d 


# Each subdirectory must supply rules for building sources it contributes
solvers/z3/%.o: ../solvers/z3/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


