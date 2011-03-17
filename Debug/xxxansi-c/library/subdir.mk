################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../xxxansi-c/library/converter.cpp 

C_SRCS += \
../xxxansi-c/library/ctype.c \
../xxxansi-c/library/getopt.c \
../xxxansi-c/library/io.c \
../xxxansi-c/library/math.c \
../xxxansi-c/library/pthread_lib.c \
../xxxansi-c/library/stdlib.c \
../xxxansi-c/library/string.c 

OBJS += \
./xxxansi-c/library/converter.o \
./xxxansi-c/library/ctype.o \
./xxxansi-c/library/getopt.o \
./xxxansi-c/library/io.o \
./xxxansi-c/library/math.o \
./xxxansi-c/library/pthread_lib.o \
./xxxansi-c/library/stdlib.o \
./xxxansi-c/library/string.o 

C_DEPS += \
./xxxansi-c/library/ctype.d \
./xxxansi-c/library/getopt.d \
./xxxansi-c/library/io.d \
./xxxansi-c/library/math.d \
./xxxansi-c/library/pthread_lib.d \
./xxxansi-c/library/stdlib.d \
./xxxansi-c/library/string.d 

CPP_DEPS += \
./xxxansi-c/library/converter.d 


# Each subdirectory must supply rules for building sources it contributes
xxxansi-c/library/%.o: ../xxxansi-c/library/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

xxxansi-c/library/%.o: ../xxxansi-c/library/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


