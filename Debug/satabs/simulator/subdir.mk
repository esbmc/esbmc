################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../satabs/simulator/concrete_counterexample.cpp \
../satabs/simulator/path_slicer.cpp \
../satabs/simulator/recurrence_solver.cpp \
../satabs/simulator/select_simulator.cpp \
../satabs/simulator/simulator_ipp.cpp \
../satabs/simulator/simulator_loop_detection.cpp \
../satabs/simulator/simulator_symex.cpp 

OBJS += \
./satabs/simulator/concrete_counterexample.o \
./satabs/simulator/path_slicer.o \
./satabs/simulator/recurrence_solver.o \
./satabs/simulator/select_simulator.o \
./satabs/simulator/simulator_ipp.o \
./satabs/simulator/simulator_loop_detection.o \
./satabs/simulator/simulator_symex.o 

CPP_DEPS += \
./satabs/simulator/concrete_counterexample.d \
./satabs/simulator/path_slicer.d \
./satabs/simulator/recurrence_solver.d \
./satabs/simulator/select_simulator.d \
./satabs/simulator/simulator_ipp.d \
./satabs/simulator/simulator_loop_detection.d \
./satabs/simulator/simulator_symex.d 


# Each subdirectory must supply rules for building sources it contributes
satabs/simulator/%.o: ../satabs/simulator/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


