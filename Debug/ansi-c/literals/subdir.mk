################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../ansi-c/literals/convert_character_literal.cpp \
../ansi-c/literals/convert_float_literal.cpp \
../ansi-c/literals/convert_integer_literal.cpp \
../ansi-c/literals/convert_string_literal.cpp \
../ansi-c/literals/parse_float.cpp \
../ansi-c/literals/unescape_string.cpp 

OBJS += \
./ansi-c/literals/convert_character_literal.o \
./ansi-c/literals/convert_float_literal.o \
./ansi-c/literals/convert_integer_literal.o \
./ansi-c/literals/convert_string_literal.o \
./ansi-c/literals/parse_float.o \
./ansi-c/literals/unescape_string.o 

CPP_DEPS += \
./ansi-c/literals/convert_character_literal.d \
./ansi-c/literals/convert_float_literal.d \
./ansi-c/literals/convert_integer_literal.d \
./ansi-c/literals/convert_string_literal.d \
./ansi-c/literals/parse_float.d \
./ansi-c/literals/unescape_string.d 


# Each subdirectory must supply rules for building sources it contributes
ansi-c/literals/%.o: ../ansi-c/literals/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


