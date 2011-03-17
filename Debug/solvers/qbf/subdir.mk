################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../solvers/qbf/qbf_quantor.o \
../solvers/qbf/qbf_qube.o \
../solvers/qbf/qbf_qube_core.o \
../solvers/qbf/qbf_skizzo.o \
../solvers/qbf/qdimacs_cnf.o \
../solvers/qbf/qdimacs_core.o 

CPP_SRCS += \
../solvers/qbf/qbf_bdd_core.cpp \
../solvers/qbf/qbf_quantor.cpp \
../solvers/qbf/qbf_qube.cpp \
../solvers/qbf/qbf_qube_core.cpp \
../solvers/qbf/qbf_skizzo.cpp \
../solvers/qbf/qbf_skizzo_core.cpp \
../solvers/qbf/qbf_squolem.cpp \
../solvers/qbf/qbf_squolem_core.cpp \
../solvers/qbf/qdimacs_cnf.cpp \
../solvers/qbf/qdimacs_core.cpp 

OBJS += \
./solvers/qbf/qbf_bdd_core.o \
./solvers/qbf/qbf_quantor.o \
./solvers/qbf/qbf_qube.o \
./solvers/qbf/qbf_qube_core.o \
./solvers/qbf/qbf_skizzo.o \
./solvers/qbf/qbf_skizzo_core.o \
./solvers/qbf/qbf_squolem.o \
./solvers/qbf/qbf_squolem_core.o \
./solvers/qbf/qdimacs_cnf.o \
./solvers/qbf/qdimacs_core.o 

CPP_DEPS += \
./solvers/qbf/qbf_bdd_core.d \
./solvers/qbf/qbf_quantor.d \
./solvers/qbf/qbf_qube.d \
./solvers/qbf/qbf_qube_core.d \
./solvers/qbf/qbf_skizzo.d \
./solvers/qbf/qbf_skizzo_core.d \
./solvers/qbf/qbf_squolem.d \
./solvers/qbf/qbf_squolem_core.d \
./solvers/qbf/qdimacs_cnf.d \
./solvers/qbf/qdimacs_core.d 


# Each subdirectory must supply rules for building sources it contributes
solvers/qbf/%.o: ../solvers/qbf/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


