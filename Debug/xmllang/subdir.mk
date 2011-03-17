################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../xmllang/lex.yy.o \
../xmllang/xml_language.o \
../xmllang/xml_parse_tree.o \
../xmllang/xml_parser.o \
../xmllang/xml_typecheck.o \
../xmllang/xmllang.o \
../xmllang/y.tab.o 

CPP_SRCS += \
../xmllang/lex.yy.cpp \
../xmllang/xml_language.cpp \
../xmllang/xml_parse_tree.cpp \
../xmllang/xml_parser.cpp \
../xmllang/xml_typecheck.cpp \
../xmllang/y.tab.cpp 

OBJS += \
./xmllang/lex.yy.o \
./xmllang/xml_language.o \
./xmllang/xml_parse_tree.o \
./xmllang/xml_parser.o \
./xmllang/xml_typecheck.o \
./xmllang/y.tab.o 

CPP_DEPS += \
./xmllang/lex.yy.d \
./xmllang/xml_language.d \
./xmllang/xml_parse_tree.d \
./xmllang/xml_parser.d \
./xmllang/xml_typecheck.d \
./xmllang/y.tab.d 


# Each subdirectory must supply rules for building sources it contributes
xmllang/%.o: ../xmllang/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


