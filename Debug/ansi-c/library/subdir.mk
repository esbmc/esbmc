################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../ansi-c/library/converter.cpp 

C_SRCS += \
../ansi-c/library/ctype.c \
../ansi-c/library/getopt.c \
../ansi-c/library/io.c \
../ansi-c/library/math.c \
../ansi-c/library/pthread_lib.c \
../ansi-c/library/stdlib.c \
../ansi-c/library/string.c \
../ansi-c/library/time.c 

OBJS += \
./ansi-c/library/converter.o \
./ansi-c/library/ctype.o \
./ansi-c/library/getopt.o \
./ansi-c/library/io.o \
./ansi-c/library/math.o \
./ansi-c/library/pthread_lib.o \
./ansi-c/library/stdlib.o \
./ansi-c/library/string.o \
./ansi-c/library/time.o 

C_DEPS += \
./ansi-c/library/ctype.d \
./ansi-c/library/getopt.d \
./ansi-c/library/io.d \
./ansi-c/library/math.d \
./ansi-c/library/pthread_lib.d \
./ansi-c/library/stdlib.d \
./ansi-c/library/string.d \
./ansi-c/library/time.d 

CPP_DEPS += \
./ansi-c/library/converter.d 


# Each subdirectory must supply rules for building sources it contributes
ansi-c/library/%.o: ../ansi-c/library/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

ansi-c/library/%.o: ../ansi-c/library/%.c
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C Compiler'
	gcc -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


