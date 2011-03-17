################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../satabs/termination/instrumentation.cpp \
../satabs/termination/rankfunction_typecheck.cpp \
../satabs/termination/ranking.cpp \
../satabs/termination/ranking_base.cpp \
../satabs/termination/ranking_qbf.cpp \
../satabs/termination/ranking_qbf_bitwise.cpp \
../satabs/termination/ranking_qbf_complete.cpp \
../satabs/termination/ranking_rankfinder.cpp \
../satabs/termination/ranking_sat.cpp \
../satabs/termination/ranking_seneschal.cpp \
../satabs/termination/ranking_tools.cpp \
../satabs/termination/termination.cpp 

OBJS += \
./satabs/termination/instrumentation.o \
./satabs/termination/rankfunction_typecheck.o \
./satabs/termination/ranking.o \
./satabs/termination/ranking_base.o \
./satabs/termination/ranking_qbf.o \
./satabs/termination/ranking_qbf_bitwise.o \
./satabs/termination/ranking_qbf_complete.o \
./satabs/termination/ranking_rankfinder.o \
./satabs/termination/ranking_sat.o \
./satabs/termination/ranking_seneschal.o \
./satabs/termination/ranking_tools.o \
./satabs/termination/termination.o 

CPP_DEPS += \
./satabs/termination/instrumentation.d \
./satabs/termination/rankfunction_typecheck.d \
./satabs/termination/ranking.d \
./satabs/termination/ranking_base.d \
./satabs/termination/ranking_qbf.d \
./satabs/termination/ranking_qbf_bitwise.d \
./satabs/termination/ranking_qbf_complete.d \
./satabs/termination/ranking_rankfinder.d \
./satabs/termination/ranking_sat.d \
./satabs/termination/ranking_seneschal.d \
./satabs/termination/ranking_tools.d \
./satabs/termination/termination.d 


# Each subdirectory must supply rules for building sources it contributes
satabs/termination/%.o: ../satabs/termination/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


