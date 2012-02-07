CXXFLAGS += -Wall -Wextra -Wno-deprecated
GCCFLAGS += -Wall -Wextra -Wno-deprecated
OPTS ?= -O3
PLATLIBS= -lgmp -ldl

# Solvers
ifdef USE_SOLVER_Z3
Z3 = $(SATDIR)/z3
endif

ifdef USE_SOLVER_BOOLECTOR
BOOLECTOR = $(SATDIR)/boolector
endif

ifdef USE_SOLVER_MINISAT1
MINISAT = $(SATDIR)/MiniSat-p_v1.14
endif

ifdef TARGET32
CXXFLAGS += -m32
GCCFLAGS += -m32
LDFLAGS += -m elf_i386
endif

ifdef STATICLINK
FINLINKTAIL+= -static
endif

ifdef LINUXCOMPAT
CC=gcc34
CXX=g++34
FINLINKBIN=g++
else
# If not in compat mode, link against libgomp. Can't do it with g++34 for
# an unknown reason.
PLATLIBS+= -lgomp
endif
