# If SATDIR isn't set yet, assume the user has solvers in their home dir.
SATDIR ?= $(HOME)

ifdef WIN_MSVC
Z31 = $(SATDIR)/z3
Z3 = $(shell cygpath -w $(Z31) | sed s/\\\\/\\\\\\\\/g )
else
Z3 = $(SATDIR)/z3
endif
Z3GCC = $(SATDIR)/z3
#BOOLECTOR = $(SATDIR)/boolector
#MINISAT = $(SATDIR)/MiniSat-p_v1.14
