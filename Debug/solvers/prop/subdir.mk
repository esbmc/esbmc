################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../solvers/prop/aig.o \
../solvers/prop/aig_formula.o \
../solvers/prop/aig_prop.o \
../solvers/prop/prop.o \
../solvers/prop/prop_conv.o \
../solvers/prop/prop_conv_store.o 

CPP_SRCS += \
../solvers/prop/aig.cpp \
../solvers/prop/aig_formula.cpp \
../solvers/prop/aig_prop.cpp \
../solvers/prop/prop.cpp \
../solvers/prop/prop_conv.cpp \
../solvers/prop/prop_conv_store.cpp 

OBJS += \
./solvers/prop/aig.o \
./solvers/prop/aig_formula.o \
./solvers/prop/aig_prop.o \
./solvers/prop/prop.o \
./solvers/prop/prop_conv.o \
./solvers/prop/prop_conv_store.o 

CPP_DEPS += \
./solvers/prop/aig.d \
./solvers/prop/aig_formula.d \
./solvers/prop/aig_prop.d \
./solvers/prop/prop.d \
./solvers/prop/prop_conv.d \
./solvers/prop/prop_conv_store.d 


# Each subdirectory must supply rules for building sources it contributes
solvers/prop/%.o: ../solvers/prop/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


