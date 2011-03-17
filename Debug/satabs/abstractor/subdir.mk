################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../satabs/abstractor/abstract_dynamic_objects.cpp \
../satabs/abstractor/abstract_expression.cpp \
../satabs/abstractor/abstract_model.cpp \
../satabs/abstractor/abstract_program.cpp \
../satabs/abstractor/abstractor.cpp \
../satabs/abstractor/abstractor_prover.cpp \
../satabs/abstractor/abstractor_satqe.cpp \
../satabs/abstractor/abstractor_wp.cpp \
../satabs/abstractor/canonicalize.cpp \
../satabs/abstractor/discover_predicates.cpp \
../satabs/abstractor/initial_abstraction.cpp \
../satabs/abstractor/lift_if.cpp \
../satabs/abstractor/predabs_aux.cpp \
../satabs/abstractor/predicate_image_prover.cpp \
../satabs/abstractor/predicate_image_satqe.cpp \
../satabs/abstractor/predicates.cpp \
../satabs/abstractor/select_abstractor.cpp 

OBJS += \
./satabs/abstractor/abstract_dynamic_objects.o \
./satabs/abstractor/abstract_expression.o \
./satabs/abstractor/abstract_model.o \
./satabs/abstractor/abstract_program.o \
./satabs/abstractor/abstractor.o \
./satabs/abstractor/abstractor_prover.o \
./satabs/abstractor/abstractor_satqe.o \
./satabs/abstractor/abstractor_wp.o \
./satabs/abstractor/canonicalize.o \
./satabs/abstractor/discover_predicates.o \
./satabs/abstractor/initial_abstraction.o \
./satabs/abstractor/lift_if.o \
./satabs/abstractor/predabs_aux.o \
./satabs/abstractor/predicate_image_prover.o \
./satabs/abstractor/predicate_image_satqe.o \
./satabs/abstractor/predicates.o \
./satabs/abstractor/select_abstractor.o 

CPP_DEPS += \
./satabs/abstractor/abstract_dynamic_objects.d \
./satabs/abstractor/abstract_expression.d \
./satabs/abstractor/abstract_model.d \
./satabs/abstractor/abstract_program.d \
./satabs/abstractor/abstractor.d \
./satabs/abstractor/abstractor_prover.d \
./satabs/abstractor/abstractor_satqe.d \
./satabs/abstractor/abstractor_wp.d \
./satabs/abstractor/canonicalize.d \
./satabs/abstractor/discover_predicates.d \
./satabs/abstractor/initial_abstraction.d \
./satabs/abstractor/lift_if.d \
./satabs/abstractor/predabs_aux.d \
./satabs/abstractor/predicate_image_prover.d \
./satabs/abstractor/predicate_image_satqe.d \
./satabs/abstractor/predicates.d \
./satabs/abstractor/select_abstractor.d 


# Each subdirectory must supply rules for building sources it contributes
satabs/abstractor/%.o: ../satabs/abstractor/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


