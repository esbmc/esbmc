################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../cvclang/cvc_language.o \
../cvclang/cvc_parser.o \
../cvclang/cvc_typecheck.o \
../cvclang/cvclang.o \
../cvclang/expr2cvc.o \
../cvclang/lex.yy.o \
../cvclang/y.tab.o 

CPP_SRCS += \
../cvclang/cvc_language.cpp \
../cvclang/cvc_parser.cpp \
../cvclang/cvc_typecheck.cpp \
../cvclang/expr2cvc.cpp \
../cvclang/lex.yy.cpp \
../cvclang/y.tab.cpp 

OBJS += \
./cvclang/cvc_language.o \
./cvclang/cvc_parser.o \
./cvclang/cvc_typecheck.o \
./cvclang/expr2cvc.o \
./cvclang/lex.yy.o \
./cvclang/y.tab.o 

CPP_DEPS += \
./cvclang/cvc_language.d \
./cvclang/cvc_parser.d \
./cvclang/cvc_typecheck.d \
./cvclang/expr2cvc.d \
./cvclang/lex.yy.d \
./cvclang/y.tab.d 


# Each subdirectory must supply rules for building sources it contributes
cvclang/%.o: ../cvclang/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


