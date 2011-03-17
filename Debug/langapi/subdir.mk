################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../langapi/ansi-c_dummy.o \
../langapi/bplang_dummy.o \
../langapi/cpp_dummy.o \
../langapi/csharp_dummy.o \
../langapi/csp_dummy.o \
../langapi/cvclang_dummy.o \
../langapi/intrep_dummy.o \
../langapi/langapi.o \
../langapi/language_ui.o \
../langapi/language_util.o \
../langapi/languages.o \
../langapi/mdllang_dummy.o \
../langapi/mode.o \
../langapi/netlist_dummy.o \
../langapi/nsf_dummy.o \
../langapi/pascal_dummy.o \
../langapi/php_dummy.o \
../langapi/promela_dummy.o \
../langapi/pvs_dummy.o \
../langapi/simplifylang_dummy.o \
../langapi/smtlang_dummy.o \
../langapi/smvlang_dummy.o \
../langapi/specc_dummy.o \
../langapi/verilog_dummy.o \
../langapi/vhdl_dummy.o 

CPP_SRCS += \
../langapi/ansi-c_dummy.cpp \
../langapi/bplang_dummy.cpp \
../langapi/cpp_dummy.cpp \
../langapi/csharp_dummy.cpp \
../langapi/csp_dummy.cpp \
../langapi/cvclang_dummy.cpp \
../langapi/intrep_dummy.cpp \
../langapi/language_ui.cpp \
../langapi/language_util.cpp \
../langapi/languages.cpp \
../langapi/mdllang_dummy.cpp \
../langapi/mode.cpp \
../langapi/netlist_dummy.cpp \
../langapi/nsf_dummy.cpp \
../langapi/pascal_dummy.cpp \
../langapi/php_dummy.cpp \
../langapi/promela_dummy.cpp \
../langapi/pvs_dummy.cpp \
../langapi/simplifylang_dummy.cpp \
../langapi/smtlang_dummy.cpp \
../langapi/smvlang_dummy.cpp \
../langapi/specc_dummy.cpp \
../langapi/verilog_dummy.cpp \
../langapi/vhdl_dummy.cpp \
../langapi/xmllang_dummy.cpp 

OBJS += \
./langapi/ansi-c_dummy.o \
./langapi/bplang_dummy.o \
./langapi/cpp_dummy.o \
./langapi/csharp_dummy.o \
./langapi/csp_dummy.o \
./langapi/cvclang_dummy.o \
./langapi/intrep_dummy.o \
./langapi/language_ui.o \
./langapi/language_util.o \
./langapi/languages.o \
./langapi/mdllang_dummy.o \
./langapi/mode.o \
./langapi/netlist_dummy.o \
./langapi/nsf_dummy.o \
./langapi/pascal_dummy.o \
./langapi/php_dummy.o \
./langapi/promela_dummy.o \
./langapi/pvs_dummy.o \
./langapi/simplifylang_dummy.o \
./langapi/smtlang_dummy.o \
./langapi/smvlang_dummy.o \
./langapi/specc_dummy.o \
./langapi/verilog_dummy.o \
./langapi/vhdl_dummy.o \
./langapi/xmllang_dummy.o 

CPP_DEPS += \
./langapi/ansi-c_dummy.d \
./langapi/bplang_dummy.d \
./langapi/cpp_dummy.d \
./langapi/csharp_dummy.d \
./langapi/csp_dummy.d \
./langapi/cvclang_dummy.d \
./langapi/intrep_dummy.d \
./langapi/language_ui.d \
./langapi/language_util.d \
./langapi/languages.d \
./langapi/mdllang_dummy.d \
./langapi/mode.d \
./langapi/netlist_dummy.d \
./langapi/nsf_dummy.d \
./langapi/pascal_dummy.d \
./langapi/php_dummy.d \
./langapi/promela_dummy.d \
./langapi/pvs_dummy.d \
./langapi/simplifylang_dummy.d \
./langapi/smtlang_dummy.d \
./langapi/smvlang_dummy.d \
./langapi/specc_dummy.d \
./langapi/verilog_dummy.d \
./langapi/vhdl_dummy.d \
./langapi/xmllang_dummy.d 


# Each subdirectory must supply rules for building sources it contributes
langapi/%.o: ../langapi/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


