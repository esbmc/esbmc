################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../goto-cc/xml_binaries/read_goto_object.cpp \
../goto-cc/xml_binaries/xml_goto_function.cpp \
../goto-cc/xml_binaries/xml_goto_function_hashing.cpp \
../goto-cc/xml_binaries/xml_goto_program.cpp \
../goto-cc/xml_binaries/xml_goto_program_hashing.cpp \
../goto-cc/xml_binaries/xml_irep_hashing.cpp \
../goto-cc/xml_binaries/xml_symbol.cpp \
../goto-cc/xml_binaries/xml_symbol_hashing.cpp 

OBJS += \
./goto-cc/xml_binaries/read_goto_object.o \
./goto-cc/xml_binaries/xml_goto_function.o \
./goto-cc/xml_binaries/xml_goto_function_hashing.o \
./goto-cc/xml_binaries/xml_goto_program.o \
./goto-cc/xml_binaries/xml_goto_program_hashing.o \
./goto-cc/xml_binaries/xml_irep_hashing.o \
./goto-cc/xml_binaries/xml_symbol.o \
./goto-cc/xml_binaries/xml_symbol_hashing.o 

CPP_DEPS += \
./goto-cc/xml_binaries/read_goto_object.d \
./goto-cc/xml_binaries/xml_goto_function.d \
./goto-cc/xml_binaries/xml_goto_function_hashing.d \
./goto-cc/xml_binaries/xml_goto_program.d \
./goto-cc/xml_binaries/xml_goto_program_hashing.d \
./goto-cc/xml_binaries/xml_irep_hashing.d \
./goto-cc/xml_binaries/xml_symbol.d \
./goto-cc/xml_binaries/xml_symbol_hashing.d 


# Each subdirectory must supply rules for building sources it contributes
goto-cc/xml_binaries/%.o: ../goto-cc/xml_binaries/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


