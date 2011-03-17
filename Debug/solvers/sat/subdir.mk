################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
O_SRCS += \
../solvers/sat/cnf.o \
../solvers/sat/cnf_clause_list.o \
../solvers/sat/dimacs_cnf.o \
../solvers/sat/pbs_dimacs_cnf.o \
../solvers/sat/read_dimacs_cnf.o \
../solvers/sat/resolution_proof.o \
../solvers/sat/satcheck.o \
../solvers/sat/satcheck_minisat.o 

CPP_SRCS += \
../solvers/sat/cnf.cpp \
../solvers/sat/cnf_clause_list.cpp \
../solvers/sat/dimacs_cnf.cpp \
../solvers/sat/pbs_dimacs_cnf.cpp \
../solvers/sat/read_dimacs_cnf.cpp \
../solvers/sat/resolution_proof.cpp \
../solvers/sat/satcheck.cpp \
../solvers/sat/satcheck_booleforce.cpp \
../solvers/sat/satcheck_limmat.cpp \
../solvers/sat/satcheck_minisat.cpp \
../solvers/sat/satcheck_minisat2.cpp \
../solvers/sat/satcheck_smvsat.cpp \
../solvers/sat/satcheck_zchaff.cpp \
../solvers/sat/satcheck_zcore.cpp 

OBJS += \
./solvers/sat/cnf.o \
./solvers/sat/cnf_clause_list.o \
./solvers/sat/dimacs_cnf.o \
./solvers/sat/pbs_dimacs_cnf.o \
./solvers/sat/read_dimacs_cnf.o \
./solvers/sat/resolution_proof.o \
./solvers/sat/satcheck.o \
./solvers/sat/satcheck_booleforce.o \
./solvers/sat/satcheck_limmat.o \
./solvers/sat/satcheck_minisat.o \
./solvers/sat/satcheck_minisat2.o \
./solvers/sat/satcheck_smvsat.o \
./solvers/sat/satcheck_zchaff.o \
./solvers/sat/satcheck_zcore.o 

CPP_DEPS += \
./solvers/sat/cnf.d \
./solvers/sat/cnf_clause_list.d \
./solvers/sat/dimacs_cnf.d \
./solvers/sat/pbs_dimacs_cnf.d \
./solvers/sat/read_dimacs_cnf.d \
./solvers/sat/resolution_proof.d \
./solvers/sat/satcheck.d \
./solvers/sat/satcheck_booleforce.d \
./solvers/sat/satcheck_limmat.d \
./solvers/sat/satcheck_minisat.d \
./solvers/sat/satcheck_minisat2.d \
./solvers/sat/satcheck_smvsat.d \
./solvers/sat/satcheck_zchaff.d \
./solvers/sat/satcheck_zcore.d 


# Each subdirectory must supply rules for building sources it contributes
solvers/sat/%.o: ../solvers/sat/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: GCC C++ Compiler'
	g++ -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o"$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


