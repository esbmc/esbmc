################################################################################
# Automatically-generated file. Do not edit!
################################################################################

-include ../makefile.init

RM := rm -rf

# All of the sources participating in the build are defined here
-include sources.mk
-include subdir.mk
-include util/subdir.mk
-include trans/subdir.mk
-include solvers/z3/subdir.mk
-include solvers/subdir.mk
-include solvers/smt/subdir.mk
-include solvers/sat/subdir.mk
-include solvers/qbf/subdir.mk
-include solvers/prop/subdir.mk
-include solvers/flattening/subdir.mk
-include solvers/dplib/subdir.mk
-include solvers/cvc/subdir.mk
-include solvers/boolector/subdir.mk
-include smvlang/subdir.mk
-include smtlang/subdir.mk
-include satqe/subdir.mk
-include promela/subdir.mk
-include pointer-analysis/subdir.mk
-include langapi/subdir.mk
-include intrep/subdir.mk
-include goto-symex/subdir.mk
-include goto-programs/subdir.mk
-include esbmc/subdir.mk
-include cvclang/subdir.mk
-include cpp/subdir.mk
-include bplang/subdir.mk
-include big-int/subdir.mk
-include ansi-c/subdir.mk
-include ansi-c/old/subdir.mk
-include ansi-c/library/subdir.mk
-include objects.mk

ifneq ($(MAKECMDGOALS),clean)
ifneq ($(strip $(C++_DEPS)),)
-include $(C++_DEPS)
endif
ifneq ($(strip $(C_DEPS)),)
-include $(C_DEPS)
endif
ifneq ($(strip $(CC_DEPS)),)
-include $(CC_DEPS)
endif
ifneq ($(strip $(CPP_DEPS)),)
-include $(CPP_DEPS)
endif
ifneq ($(strip $(CXX_DEPS)),)
-include $(CXX_DEPS)
endif
ifneq ($(strip $(C_UPPER_DEPS)),)
-include $(C_UPPER_DEPS)
endif
endif

-include ../makefile.defs

# Add inputs and outputs from these tool invocations to the build variables 

# All Target
all: cprover

# Tool invocations
cprover: $(OBJS) $(USER_OBJS)
	@echo 'Building target: $@'
	@echo 'Invoking: GCC C++ Linker'
	g++  -o"cprover" $(OBJS) $(USER_OBJS) $(LIBS)
	@echo 'Finished building target: $@'
	@echo ' '

# Other Targets
clean:
	-$(RM) $(OBJS)$(C++_DEPS)$(C_DEPS)$(CC_DEPS)$(CPP_DEPS)$(EXECUTABLES)$(CXX_DEPS)$(C_UPPER_DEPS) cprover
	-@echo ' '

.PHONY: all clean dependents
.SECONDARY:

-include ../makefile.targets
