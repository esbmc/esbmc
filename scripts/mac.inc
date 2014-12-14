CXXFLAGS += -Wall -Wextra -Wno-deprecated -DONAMAC -stdlib=libc++
GCCFLAGS += -Wall -Wextra -Wno-deprecated -DONAMAC
OPTS ?= -O3
PLATLIBS=

# Solvers
ifdef USE_SOLVER_Z3
Z3 = $(SATDIR)/z3
CXXFLAGS+= -DZ3
Z3_SOLVER_LIBS+= -L$(Z3)/lib -lz3
INCLUDES+= -I $(Z3)/include
endif

ifdef USE_SOLVER_BOOLECTOR
BOOLECTOR = $(SATDIR)/boolector-2.0.1
CXXFLAGS+= -DBOOLECTOR
SOLVER_LIBS+= -L$(BOOLECTOR)/boolector -lboolector -L$(BOOLECTOR)/lingeling -llgl
INCLUDES+= -I $(BOOLECTOR)/boolector
endif

ifdef USE_SOLVER_MINISAT
MINISAT = $(MINISAT_PATH)
CXXFLAGS+= -DMINISAT
LIBS+=$(MINISAT_PATH)/core/lib_release.a
endif

# Other goo
FULLUNAME:=$(shell uname -a)
WEARE64BIT:=$(shell uname -a | egrep "(CYGWIN.*WOW64|x86_64)")
ifneq ($(WEARE64BIT),)
CXXFLAGS += -m64
GCCFLAGS += -m64
else
CXXFLAGS += -m32
GCCFLAGS += -m32
endif

ifdef STATICLINK
$(error Can't statically link anything on a mac)
endif

PLATLIBS+= -lpthread

SOLVER_LIBS+=$(Z3_SOLVER_LIBS)


CC=     clang
CXX=    clang++
DEPCXX= clang++

HOSTCC=     clang
HOSTCXX=    clang++
HOSTDEPCXX= clang++
HOSTLD=     ld
HOSTAR=     ar
HOPTBLOBGENLD= ld
HOSTPARTLINKLD= ld
HOSTPARTLINKOUT= -o
HOSTPARTLINKOPTS= -r $(LDFLAGS)
HOSTFINLINKBIN= $(CXX)
HOSTFINLINKOUT= -o
HOSTFINLINKTAIL= $(CXXFLAGS) $(LIBS) $(PLATLIBS)
HOSTCFLAGS=$(CFLAGS)
HOSTCXXFLAGS=$(CXXFLAGS)
HOSTLDFLAGS=$(LDFLAGS)
