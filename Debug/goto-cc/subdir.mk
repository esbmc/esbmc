################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../goto-cc/cmdline_options.cpp \
../goto-cc/compile.cpp \
../goto-cc/gcc_cmdline.cpp \
../goto-cc/goto-cc.cpp 

OBJS += \
./goto-cc/cmdline_options.o \
./goto-cc/compile.o \
./goto-cc/gcc_cmdline.o \
./goto-cc/goto-cc.o 

CPP_DEPS += \
./goto-cc/cmdline_options.d \
./goto-cc/compile.d \
./goto-cc/gcc_cmdline.d \
./goto-cc/goto-cc.d 


# Each subdirectory must supply rules for building sources it contributes
goto-cc/%.o: ../goto-cc/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


