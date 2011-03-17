################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../satabs/refiner/add_invariants.cpp \
../satabs/refiner/refiner.cpp \
../satabs/refiner/refiner_ipp.cpp \
../satabs/refiner/refiner_lifter.cpp \
../satabs/refiner/refiner_wp.cpp \
../satabs/refiner/refiner_wp_async.cpp \
../satabs/refiner/select_refiner.cpp \
../satabs/refiner/substitute_invariants.cpp \
../satabs/refiner/trans_wp.cpp \
../satabs/refiner/transition_cache.cpp \
../satabs/refiner/transition_refiner.cpp 

OBJS += \
./satabs/refiner/add_invariants.o \
./satabs/refiner/refiner.o \
./satabs/refiner/refiner_ipp.o \
./satabs/refiner/refiner_lifter.o \
./satabs/refiner/refiner_wp.o \
./satabs/refiner/refiner_wp_async.o \
./satabs/refiner/select_refiner.o \
./satabs/refiner/substitute_invariants.o \
./satabs/refiner/trans_wp.o \
./satabs/refiner/transition_cache.o \
./satabs/refiner/transition_refiner.o 

CPP_DEPS += \
./satabs/refiner/add_invariants.d \
./satabs/refiner/refiner.d \
./satabs/refiner/refiner_ipp.d \
./satabs/refiner/refiner_lifter.d \
./satabs/refiner/refiner_wp.d \
./satabs/refiner/refiner_wp_async.d \
./satabs/refiner/select_refiner.d \
./satabs/refiner/substitute_invariants.d \
./satabs/refiner/trans_wp.d \
./satabs/refiner/transition_cache.d \
./satabs/refiner/transition_refiner.d 


# Each subdirectory must supply rules for building sources it contributes
satabs/refiner/%.o: ../satabs/refiner/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


