################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../smvlang/expr2smv.o \
../smvlang/lex.yy.o \
../smvlang/smv_language.o \
../smvlang/smv_parse_tree.o \
../smvlang/smv_parser.o \
../smvlang/smv_typecheck.o \
../smvlang/smvlang.o \
../smvlang/y.tab.o 

CPP_SRCS += \
../smvlang/expr2smv.cpp \
../smvlang/lex.yy.cpp \
../smvlang/smv_language.cpp \
../smvlang/smv_parse_tree.cpp \
../smvlang/smv_parser.cpp \
../smvlang/smv_typecheck.cpp \
../smvlang/y.tab.cpp 

OBJS += \
./smvlang/expr2smv.o \
./smvlang/lex.yy.o \
./smvlang/smv_language.o \
./smvlang/smv_parse_tree.o \
./smvlang/smv_parser.o \
./smvlang/smv_typecheck.o \
./smvlang/y.tab.o 

CPP_DEPS += \
./smvlang/expr2smv.d \
./smvlang/lex.yy.d \
./smvlang/smv_language.d \
./smvlang/smv_parse_tree.d \
./smvlang/smv_parser.d \
./smvlang/smv_typecheck.d \
./smvlang/y.tab.d 


# Each subdirectory must supply rules for building sources it contributes
smvlang/%.o: ../smvlang/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


