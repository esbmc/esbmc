################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../goto-symex/basic_symex.o \
../goto-symex/build_goto_trace.o \
../goto-symex/builtin_functions.o \
../goto-symex/dynamic_allocation.o \
../goto-symex/execution_state.o \
../goto-symex/goto-symex.o \
../goto-symex/goto_symex_state.o \
../goto-symex/goto_trace.o \
../goto-symex/postcondition.o \
../goto-symex/precondition.o \
../goto-symex/reachability_tree.o \
../goto-symex/read_write_set.o \
../goto-symex/slice.o \
../goto-symex/slice_by_trace.o \
../goto-symex/symex_dereference.o \
../goto-symex/symex_function.o \
../goto-symex/symex_goto.o \
../goto-symex/symex_main.o \
../goto-symex/symex_other.o \
../goto-symex/symex_target.o \
../goto-symex/symex_target_equation.o \
../goto-symex/symex_valid_object.o \
../goto-symex/xml_goto_trace.o 

CPP_SRCS += \
../goto-symex/basic_symex.cpp \
../goto-symex/build_goto_trace.cpp \
../goto-symex/builtin_functions.cpp \
../goto-symex/dynamic_allocation.cpp \
../goto-symex/execution_state.cpp \
../goto-symex/goto_symex_state.cpp \
../goto-symex/goto_trace.cpp \
../goto-symex/postcondition.cpp \
../goto-symex/precondition.cpp \
../goto-symex/reachability_tree.cpp \
../goto-symex/read_write_set.cpp \
../goto-symex/slice.cpp \
../goto-symex/slice_by_trace.cpp \
../goto-symex/symex_dereference.cpp \
../goto-symex/symex_function.cpp \
../goto-symex/symex_goto.cpp \
../goto-symex/symex_main.cpp \
../goto-symex/symex_other.cpp \
../goto-symex/symex_target.cpp \
../goto-symex/symex_target_equation.cpp \
../goto-symex/symex_valid_object.cpp \
../goto-symex/xml_goto_trace.cpp 

OBJS += \
./goto-symex/basic_symex.o \
./goto-symex/build_goto_trace.o \
./goto-symex/builtin_functions.o \
./goto-symex/dynamic_allocation.o \
./goto-symex/execution_state.o \
./goto-symex/goto_symex_state.o \
./goto-symex/goto_trace.o \
./goto-symex/postcondition.o \
./goto-symex/precondition.o \
./goto-symex/reachability_tree.o \
./goto-symex/read_write_set.o \
./goto-symex/slice.o \
./goto-symex/slice_by_trace.o \
./goto-symex/symex_dereference.o \
./goto-symex/symex_function.o \
./goto-symex/symex_goto.o \
./goto-symex/symex_main.o \
./goto-symex/symex_other.o \
./goto-symex/symex_target.o \
./goto-symex/symex_target_equation.o \
./goto-symex/symex_valid_object.o \
./goto-symex/xml_goto_trace.o 

CPP_DEPS += \
./goto-symex/basic_symex.d \
./goto-symex/build_goto_trace.d \
./goto-symex/builtin_functions.d \
./goto-symex/dynamic_allocation.d \
./goto-symex/execution_state.d \
./goto-symex/goto_symex_state.d \
./goto-symex/goto_trace.d \
./goto-symex/postcondition.d \
./goto-symex/precondition.d \
./goto-symex/reachability_tree.d \
./goto-symex/read_write_set.d \
./goto-symex/slice.d \
./goto-symex/slice_by_trace.d \
./goto-symex/symex_dereference.d \
./goto-symex/symex_function.d \
./goto-symex/symex_goto.d \
./goto-symex/symex_main.d \
./goto-symex/symex_other.d \
./goto-symex/symex_target.d \
./goto-symex/symex_target_equation.d \
./goto-symex/symex_valid_object.d \
./goto-symex/xml_goto_trace.d 


# Each subdirectory must supply rules for building sources it contributes
goto-symex/%.o: ../goto-symex/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


