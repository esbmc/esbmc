# If SATDIR isn't set yet, assume the user has solvers in their home dir.
SATDIR ?= $(HOME)

Z31 = $(SATDIR)/z3
Z3 = $(shell cygpath -w $(Z31) | sed s/\\\\/\\\\\\\\/g )
#BOOLECTOR = $(SATDIR)/boolector
#MINISAT = $(SATDIR)/MiniSat-p_v1.14
