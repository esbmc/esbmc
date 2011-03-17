################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../bplang/bp_language.o \
../bplang/bp_parse_tree.o \
../bplang/bp_parser.o \
../bplang/bp_typecheck.o \
../bplang/bp_typecheck_code.o \
../bplang/bp_typecheck_expr.o \
../bplang/bp_util.o \
../bplang/bplang.o \
../bplang/expr2bp.o \
../bplang/lex.yy.o \
../bplang/y.tab.o 

CPP_SRCS += \
../bplang/bp_language.cpp \
../bplang/bp_parse_tree.cpp \
../bplang/bp_parser.cpp \
../bplang/bp_typecheck.cpp \
../bplang/bp_typecheck_code.cpp \
../bplang/bp_typecheck_expr.cpp \
../bplang/bp_util.cpp \
../bplang/expr2bp.cpp \
../bplang/lex.yy.cpp \
../bplang/y.tab.cpp 

OBJS += \
./bplang/bp_language.o \
./bplang/bp_parse_tree.o \
./bplang/bp_parser.o \
./bplang/bp_typecheck.o \
./bplang/bp_typecheck_code.o \
./bplang/bp_typecheck_expr.o \
./bplang/bp_util.o \
./bplang/expr2bp.o \
./bplang/lex.yy.o \
./bplang/y.tab.o 

CPP_DEPS += \
./bplang/bp_language.d \
./bplang/bp_parse_tree.d \
./bplang/bp_parser.d \
./bplang/bp_typecheck.d \
./bplang/bp_typecheck_code.d \
./bplang/bp_typecheck_expr.d \
./bplang/bp_util.d \
./bplang/expr2bp.d \
./bplang/lex.yy.d \
./bplang/y.tab.d 


# Each subdirectory must supply rules for building sources it contributes
bplang/%.o: ../bplang/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


