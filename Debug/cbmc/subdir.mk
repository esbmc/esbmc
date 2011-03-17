################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../cbmc/bmc.o \
../cbmc/boolector.o \
../cbmc/bv_cbmc.o \
../cbmc/counterexample_beautification.o \
../cbmc/counterexample_beautification_greedy.o \
../cbmc/counterexample_beautification_pbs.o \
../cbmc/cvc.o \
../cbmc/dimacs.o \
../cbmc/document_subgoals.o \
../cbmc/main.o \
../cbmc/parseoptions.o \
../cbmc/show_vcc.o \
../cbmc/smt.o \
../cbmc/symex_bmc.o \
../cbmc/z3.o 

CPP_SRCS += \
../cbmc/bmc.cpp \
../cbmc/boolector.cpp \
../cbmc/bv_cbmc.cpp \
../cbmc/counterexample_beautification.cpp \
../cbmc/counterexample_beautification_greedy.cpp \
../cbmc/counterexample_beautification_pbs.cpp \
../cbmc/cvc.cpp \
../cbmc/dimacs.cpp \
../cbmc/document_subgoals.cpp \
../cbmc/main.cpp \
../cbmc/parseoptions.cpp \
../cbmc/show_vcc.cpp \
../cbmc/smt.cpp \
../cbmc/symex_bmc.cpp \
../cbmc/z3.cpp 

OBJS += \
./cbmc/bmc.o \
./cbmc/boolector.o \
./cbmc/bv_cbmc.o \
./cbmc/counterexample_beautification.o \
./cbmc/counterexample_beautification_greedy.o \
./cbmc/counterexample_beautification_pbs.o \
./cbmc/cvc.o \
./cbmc/dimacs.o \
./cbmc/document_subgoals.o \
./cbmc/main.o \
./cbmc/parseoptions.o \
./cbmc/show_vcc.o \
./cbmc/smt.o \
./cbmc/symex_bmc.o \
./cbmc/z3.o 

CPP_DEPS += \
./cbmc/bmc.d \
./cbmc/boolector.d \
./cbmc/bv_cbmc.d \
./cbmc/counterexample_beautification.d \
./cbmc/counterexample_beautification_greedy.d \
./cbmc/counterexample_beautification_pbs.d \
./cbmc/cvc.d \
./cbmc/dimacs.d \
./cbmc/document_subgoals.d \
./cbmc/main.d \
./cbmc/parseoptions.d \
./cbmc/show_vcc.d \
./cbmc/smt.d \
./cbmc/symex_bmc.d \
./cbmc/z3.d 


# Each subdirectory must supply rules for building sources it contributes
cbmc/%.o: ../cbmc/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


