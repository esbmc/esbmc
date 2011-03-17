################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../xxxansi-c/old/c_code.cpp \
../xxxansi-c/old/c_expr.cpp \
../xxxansi-c/old/c_type.cpp \
../xxxansi-c/old/c_typecheck.cpp \
../xxxansi-c/old/convert-c.cpp 

OBJS += \
./xxxansi-c/old/c_code.o \
./xxxansi-c/old/c_expr.o \
./xxxansi-c/old/c_type.o \
./xxxansi-c/old/c_typecheck.o \
./xxxansi-c/old/convert-c.o 

CPP_DEPS += \
./xxxansi-c/old/c_code.d \
./xxxansi-c/old/c_expr.d \
./xxxansi-c/old/c_type.d \
./xxxansi-c/old/c_typecheck.d \
./xxxansi-c/old/convert-c.d 


# Each subdirectory must supply rules for building sources it contributes
xxxansi-c/old/%.o: ../xxxansi-c/old/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


