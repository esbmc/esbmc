CXXFLAGS += -Wall -Wextra -Wno-deprecated
GCCFLAGS += -Wall -Wextra -Wno-deprecated
OPTS ?= -O3
PLATLIBS= -lgmp -ldl

# Solvers
ifdef USE_SOLVER_Z3
Z3 = $(SATDIR)/z3
CXXFLAGS+= -DZ3
Z3_SOLVER_LIBS+= -L$(Z3)/lib -lz3
INCLUDES+= -I $(Z3)/include
endif

ifdef USE_SOLVER_BOOLECTOR
BOOLECTOR = $(SATDIR)/boolector
CXXFLAGS+= -DBOOLECTOR
SOLVER_LIBS+= -L$(BOOLECTOR)/lib -lboolector
endif

ifdef USE_SOLVER_MINISAT1
MINISAT = $(SATDIR)/MiniSat-p_v1.14
CXXFLAGS+= -DMINISAT
endif

# Other goo

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
FINLINKTAIL+= -Wl,--defsym=memcpy=memmove
STATICLIBCPPPATH:=$(shell g++ -print-file-name=libstdc++.a)
LIBS+= $(STATICLIBCPPPATH)

# Link statically against Z3.
ifdef USE_SOLVER_Z3
Z3_SOLVER_LIBS= $(Z3)/lib/libz3.a
endif

else
# If not in compat mode, link against libgomp. Can't do it with g++34 for
# an unknown reason.
PLATLIBS+= -lgomp
endif

SOLVER_LIBS+=$(Z3_SOLVER_LIBS)
