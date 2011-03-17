################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../ansi-c/old/c_code.cpp \
../ansi-c/old/c_expr.cpp \
../ansi-c/old/c_type.cpp \
../ansi-c/old/c_typecheck.cpp \
../ansi-c/old/convert-c.cpp 

OBJS += \
./ansi-c/old/c_code.o \
./ansi-c/old/c_expr.o \
./ansi-c/old/c_type.o \
./ansi-c/old/c_typecheck.o \
./ansi-c/old/convert-c.o 

CPP_DEPS += \
./ansi-c/old/c_code.d \
./ansi-c/old/c_expr.d \
./ansi-c/old/c_type.d \
./ansi-c/old/c_typecheck.d \
./ansi-c/old/convert-c.d 


# Each subdirectory must supply rules for building sources it contributes
ansi-c/old/%.o: ../ansi-c/old/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


