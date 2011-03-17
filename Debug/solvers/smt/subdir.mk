################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../solvers/smt/smt_conv.o \
../solvers/smt/smt_dec.o \
../solvers/smt/smt_prop.o 

CPP_SRCS += \
../solvers/smt/smt_conv.cpp \
../solvers/smt/smt_dec.cpp \
../solvers/smt/smt_prop.cpp 

OBJS += \
./solvers/smt/smt_conv.o \
./solvers/smt/smt_dec.o \
./solvers/smt/smt_prop.o 

CPP_DEPS += \
./solvers/smt/smt_conv.d \
./solvers/smt/smt_dec.d \
./solvers/smt/smt_prop.d 


# Each subdirectory must supply rules for building sources it contributes
solvers/smt/%.o: ../solvers/smt/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


