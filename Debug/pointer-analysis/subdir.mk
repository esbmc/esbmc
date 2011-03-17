################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../pointer-analysis/add_failed_symbols.o \
../pointer-analysis/dereference.o \
../pointer-analysis/goto_program_dereference.o \
../pointer-analysis/pointer-analysis.o \
../pointer-analysis/pointer_offset_sum.o \
../pointer-analysis/show_value_sets.o \
../pointer-analysis/value_set.o \
../pointer-analysis/value_set_analysis.o \
../pointer-analysis/value_set_analysis_fi.o \
../pointer-analysis/value_set_analysis_fivr.o \
../pointer-analysis/value_set_analysis_fivrns.o \
../pointer-analysis/value_set_domain.o \
../pointer-analysis/value_set_domain_fi.o \
../pointer-analysis/value_set_domain_fivr.o \
../pointer-analysis/value_set_domain_fivrns.o \
../pointer-analysis/value_set_fi.o \
../pointer-analysis/value_set_fivr.o \
../pointer-analysis/value_set_fivrns.o 

CPP_SRCS += \
../pointer-analysis/add_failed_symbols.cpp \
../pointer-analysis/dereference.cpp \
../pointer-analysis/goto_program_dereference.cpp \
../pointer-analysis/pointer_offset_sum.cpp \
../pointer-analysis/show_value_sets.cpp \
../pointer-analysis/value_propagation.cpp \
../pointer-analysis/value_set.cpp \
../pointer-analysis/value_set_analysis.cpp \
../pointer-analysis/value_set_analysis_fi.cpp \
../pointer-analysis/value_set_analysis_fivr.cpp \
../pointer-analysis/value_set_analysis_fivrns.cpp \
../pointer-analysis/value_set_domain.cpp \
../pointer-analysis/value_set_domain_fi.cpp \
../pointer-analysis/value_set_domain_fivr.cpp \
../pointer-analysis/value_set_domain_fivrns.cpp \
../pointer-analysis/value_set_fi.cpp \
../pointer-analysis/value_set_fivr.cpp \
../pointer-analysis/value_set_fivrns.cpp 

OBJS += \
./pointer-analysis/add_failed_symbols.o \
./pointer-analysis/dereference.o \
./pointer-analysis/goto_program_dereference.o \
./pointer-analysis/pointer_offset_sum.o \
./pointer-analysis/show_value_sets.o \
./pointer-analysis/value_propagation.o \
./pointer-analysis/value_set.o \
./pointer-analysis/value_set_analysis.o \
./pointer-analysis/value_set_analysis_fi.o \
./pointer-analysis/value_set_analysis_fivr.o \
./pointer-analysis/value_set_analysis_fivrns.o \
./pointer-analysis/value_set_domain.o \
./pointer-analysis/value_set_domain_fi.o \
./pointer-analysis/value_set_domain_fivr.o \
./pointer-analysis/value_set_domain_fivrns.o \
./pointer-analysis/value_set_fi.o \
./pointer-analysis/value_set_fivr.o \
./pointer-analysis/value_set_fivrns.o 

CPP_DEPS += \
./pointer-analysis/add_failed_symbols.d \
./pointer-analysis/dereference.d \
./pointer-analysis/goto_program_dereference.d \
./pointer-analysis/pointer_offset_sum.d \
./pointer-analysis/show_value_sets.d \
./pointer-analysis/value_propagation.d \
./pointer-analysis/value_set.d \
./pointer-analysis/value_set_analysis.d \
./pointer-analysis/value_set_analysis_fi.d \
./pointer-analysis/value_set_analysis_fivr.d \
./pointer-analysis/value_set_analysis_fivrns.d \
./pointer-analysis/value_set_domain.d \
./pointer-analysis/value_set_domain_fi.d \
./pointer-analysis/value_set_domain_fivr.d \
./pointer-analysis/value_set_domain_fivrns.d \
./pointer-analysis/value_set_fi.d \
./pointer-analysis/value_set_fivr.d \
./pointer-analysis/value_set_fivrns.d 


# Each subdirectory must supply rules for building sources it contributes
pointer-analysis/%.o: ../pointer-analysis/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


