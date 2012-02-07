CXXFLAGS += -Wall -Wextra -Wno-deprecated
GCCFLAGS += -Wall -Wextra -Wno-deprecated
OPTS ?= -O3
PLATLIBS= -lgmp -ldl

# Solvers
Z3 = $(SATDIR)/z3
BOOLECTOR = $(SATDIR)/boolector
MINISAT = $(SATDIR)/MiniSat-p_v1.14

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
