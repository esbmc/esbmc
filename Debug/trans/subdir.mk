################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../trans/bmc_map.cpp \
../trans/compute_ct.cpp \
../trans/counterexample.cpp \
../trans/get_trans.cpp \
../trans/instantiate.cpp \
../trans/ldg.cpp \
../trans/map_aigs.cpp \
../trans/netlist.cpp \
../trans/netlist_trans.cpp \
../trans/property.cpp \
../trans/show_modules.cpp \
../trans/trans_trace.cpp \
../trans/unwind.cpp \
../trans/unwind_netlist.cpp \
../trans/var_map.cpp \
../trans/word_level_trans.cpp 

OBJS += \
./trans/bmc_map.o \
./trans/compute_ct.o \
./trans/counterexample.o \
./trans/get_trans.o \
./trans/instantiate.o \
./trans/ldg.o \
./trans/map_aigs.o \
./trans/netlist.o \
./trans/netlist_trans.o \
./trans/property.o \
./trans/show_modules.o \
./trans/trans_trace.o \
./trans/unwind.o \
./trans/unwind_netlist.o \
./trans/var_map.o \
./trans/word_level_trans.o 

CPP_DEPS += \
./trans/bmc_map.d \
./trans/compute_ct.d \
./trans/counterexample.d \
./trans/get_trans.d \
./trans/instantiate.d \
./trans/ldg.d \
./trans/map_aigs.d \
./trans/netlist.d \
./trans/netlist_trans.d \
./trans/property.d \
./trans/show_modules.d \
./trans/trans_trace.d \
./trans/unwind.d \
./trans/unwind_netlist.d \
./trans/var_map.d \
./trans/word_level_trans.d 


# Each subdirectory must supply rules for building sources it contributes
trans/%.o: ../trans/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


