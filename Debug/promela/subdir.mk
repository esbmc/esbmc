################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../promela/expr2promela.cpp \
../promela/promela_language.cpp \
../promela/promela_parse_tree.cpp \
../promela/promela_parser.cpp \
../promela/promela_typecheck.cpp 

OBJS += \
./promela/expr2promela.o \
./promela/promela_language.o \
./promela/promela_parse_tree.o \
./promela/promela_parser.o \
./promela/promela_typecheck.o 

CPP_DEPS += \
./promela/expr2promela.d \
./promela/promela_language.d \
./promela/promela_parse_tree.d \
./promela/promela_parser.d \
./promela/promela_typecheck.d 


# Each subdirectory must supply rules for building sources it contributes
promela/%.o: ../promela/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


