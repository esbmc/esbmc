################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../smtlang/builtin_theories.cpp \
../smtlang/expr2smt.cpp \
../smtlang/lex.yysmt.cpp \
../smtlang/smt_finalizer.cpp \
../smtlang/smt_finalizer_AUFLIA.cpp \
../smtlang/smt_finalizer_AUFLIRA.cpp \
../smtlang/smt_finalizer_AUFNIRA.cpp \
../smtlang/smt_finalizer_QF_IDL.cpp \
../smtlang/smt_finalizer_QF_LIA.cpp \
../smtlang/smt_finalizer_QF_LRA.cpp \
../smtlang/smt_finalizer_QF_UFBV32.cpp \
../smtlang/smt_finalizer_generic.cpp \
../smtlang/smt_language.cpp \
../smtlang/smt_link.cpp \
../smtlang/smt_logics.cpp \
../smtlang/smt_parse_tree.cpp \
../smtlang/smt_parser.cpp \
../smtlang/smt_strings.cpp \
../smtlang/smt_typecheck.cpp \
../smtlang/smt_typecheck_expr.cpp 

OBJS += \
./smtlang/builtin_theories.o \
./smtlang/expr2smt.o \
./smtlang/lex.yysmt.o \
./smtlang/smt_finalizer.o \
./smtlang/smt_finalizer_AUFLIA.o \
./smtlang/smt_finalizer_AUFLIRA.o \
./smtlang/smt_finalizer_AUFNIRA.o \
./smtlang/smt_finalizer_QF_IDL.o \
./smtlang/smt_finalizer_QF_LIA.o \
./smtlang/smt_finalizer_QF_LRA.o \
./smtlang/smt_finalizer_QF_UFBV32.o \
./smtlang/smt_finalizer_generic.o \
./smtlang/smt_language.o \
./smtlang/smt_link.o \
./smtlang/smt_logics.o \
./smtlang/smt_parse_tree.o \
./smtlang/smt_parser.o \
./smtlang/smt_strings.o \
./smtlang/smt_typecheck.o \
./smtlang/smt_typecheck_expr.o 

CPP_DEPS += \
./smtlang/builtin_theories.d \
./smtlang/expr2smt.d \
./smtlang/lex.yysmt.d \
./smtlang/smt_finalizer.d \
./smtlang/smt_finalizer_AUFLIA.d \
./smtlang/smt_finalizer_AUFLIRA.d \
./smtlang/smt_finalizer_AUFNIRA.d \
./smtlang/smt_finalizer_QF_IDL.d \
./smtlang/smt_finalizer_QF_LIA.d \
./smtlang/smt_finalizer_QF_LRA.d \
./smtlang/smt_finalizer_QF_UFBV32.d \
./smtlang/smt_finalizer_generic.d \
./smtlang/smt_language.d \
./smtlang/smt_link.d \
./smtlang/smt_logics.d \
./smtlang/smt_parse_tree.d \
./smtlang/smt_parser.d \
./smtlang/smt_strings.d \
./smtlang/smt_typecheck.d \
./smtlang/smt_typecheck_expr.d 


# Each subdirectory must supply rules for building sources it contributes
smtlang/%.o: ../smtlang/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


