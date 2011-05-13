CC     = gcc
CXX    = g++
LD     = ld
AR     = ar
YACC   = bison -y
YFLAGS = -v
LEX    = flex
#CC     = icc
#CXX  = icpc
#USE_AR = 1

# Linux etc.
#GCCFLAGS = -Wall -MMD -O2 -Wno-deprecated # -ggdb -O0
GCCFLAGS = -Wall -MMD -Wextra -O3 -finline-limit=1000000 -Wno-deprecated -pg
LDFLAGS=

# use these for Cygwin:
# GCCFLAGS = -Wall -MMD -O2 -mno-cygwin
# LDFLAGS = -mno-cygwin

# SMT and SAT solver locations. Put your own machine-specific absolute paths to
# these in "local.inc". Z3, BOOLECTOR and MINISAT are currently required.
#Z3=
#BOOLECTOR=
#CHAFF=
#BOOLEFORCE=
#MINISAT=
#MINISAT2=
#SMVSAT=

LANGUAGES = ansi-c csp intrep pvs smvlang verilog vhdl netlist cpp specc \
            pascal bplang simplifylang cvclang csharp \
            smtlang nsf mdllang

# modules that we have
##MODULE_PROVER = 1
##MODULE_IPP = 1
#MODULE_INTERPOLATION = 1
#MODULE_BV_REFINEMENT = 1
##MODULE_SATQE = 1
##MODULE_CPP = 1
##MODULE_SMTLANG = 1
##MODULE_SPECC = 1
##MODULE_VERILOG = 1
##MODULE_VHDL = 1
##MODULE_PASCAL = 1
#MODULE_PHP = 1
##MODULE_SIMPLIFYLANG = 1
##MODULE_HW_CBMC = 1
##MODULE_CEMC = 1
