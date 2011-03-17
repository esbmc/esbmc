################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../satabs/modelchecker/abstract_counterexample.cpp \
../satabs/modelchecker/abstract_state.cpp \
../satabs/modelchecker/modelchecker.cpp \
../satabs/modelchecker/modelchecker_boolean_program.cpp \
../satabs/modelchecker/modelchecker_smv.cpp \
../satabs/modelchecker/modelchecker_spin.cpp \
../satabs/modelchecker/select_modelchecker.cpp 

OBJS += \
./satabs/modelchecker/abstract_counterexample.o \
./satabs/modelchecker/abstract_state.o \
./satabs/modelchecker/modelchecker.o \
./satabs/modelchecker/modelchecker_boolean_program.o \
./satabs/modelchecker/modelchecker_smv.o \
./satabs/modelchecker/modelchecker_spin.o \
./satabs/modelchecker/select_modelchecker.o 

CPP_DEPS += \
./satabs/modelchecker/abstract_counterexample.d \
./satabs/modelchecker/abstract_state.d \
./satabs/modelchecker/modelchecker.d \
./satabs/modelchecker/modelchecker_boolean_program.d \
./satabs/modelchecker/modelchecker_smv.d \
./satabs/modelchecker/modelchecker_spin.d \
./satabs/modelchecker/select_modelchecker.d 


# Each subdirectory must supply rules for building sources it contributes
satabs/modelchecker/%.o: ../satabs/modelchecker/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


