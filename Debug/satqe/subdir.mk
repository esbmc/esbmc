################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../satqe/cube_set.cpp \
../satqe/cubes.cpp \
../satqe/sat_cubes.cpp \
../satqe/satqe_satcheck.cpp 

OBJS += \
./satqe/cube_set.o \
./satqe/cubes.o \
./satqe/sat_cubes.o \
./satqe/satqe_satcheck.o 

CPP_DEPS += \
./satqe/cube_set.d \
./satqe/cubes.d \
./satqe/sat_cubes.d \
./satqe/satqe_satcheck.d 


# Each subdirectory must supply rules for building sources it contributes
satqe/%.o: ../satqe/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


