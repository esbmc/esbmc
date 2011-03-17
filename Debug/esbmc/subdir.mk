################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../esbmc/bmc.o \
../esbmc/boolector.o \
../esbmc/bv_cbmc.o \
../esbmc/counterexample_beautification.o \
../esbmc/counterexample_beautification_greedy.o \
../esbmc/counterexample_beautification_pbs.o \
../esbmc/cvc.o \
../esbmc/dimacs.o \
../esbmc/document_subgoals.o \
../esbmc/main.o \
../esbmc/parseoptions.o \
../esbmc/show_vcc.o \
../esbmc/smt.o \
../esbmc/symex_bmc.o \
../esbmc/z3.o 

CPP_SRCS += \
../esbmc/bmc.cpp \
../esbmc/boolector.cpp \
../esbmc/bv_cbmc.cpp \
../esbmc/counterexample_beautification.cpp \
../esbmc/counterexample_beautification_greedy.cpp \
../esbmc/counterexample_beautification_pbs.cpp \
../esbmc/cvc.cpp \
../esbmc/dimacs.cpp \
../esbmc/document_subgoals.cpp \
../esbmc/main.cpp \
../esbmc/parseoptions.cpp \
../esbmc/show_vcc.cpp \
../esbmc/smt.cpp \
../esbmc/symex_bmc.cpp \
../esbmc/z3.cpp 

OBJS += \
./esbmc/bmc.o \
./esbmc/boolector.o \
./esbmc/bv_cbmc.o \
./esbmc/counterexample_beautification.o \
./esbmc/counterexample_beautification_greedy.o \
./esbmc/counterexample_beautification_pbs.o \
./esbmc/cvc.o \
./esbmc/dimacs.o \
./esbmc/document_subgoals.o \
./esbmc/main.o \
./esbmc/parseoptions.o \
./esbmc/show_vcc.o \
./esbmc/smt.o \
./esbmc/symex_bmc.o \
./esbmc/z3.o 

CPP_DEPS += \
./esbmc/bmc.d \
./esbmc/boolector.d \
./esbmc/bv_cbmc.d \
./esbmc/counterexample_beautification.d \
./esbmc/counterexample_beautification_greedy.d \
./esbmc/counterexample_beautification_pbs.d \
./esbmc/cvc.d \
./esbmc/dimacs.d \
./esbmc/document_subgoals.d \
./esbmc/main.d \
./esbmc/parseoptions.d \
./esbmc/show_vcc.d \
./esbmc/smt.d \
./esbmc/symex_bmc.d \
./esbmc/z3.d 


# Each subdirectory must supply rules for building sources it contributes
esbmc/%.o: ../esbmc/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


