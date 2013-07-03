OPTS ?= -O3
PLATLIBS= -lgmp -ldl -lgcc

# Solvers
ifdef USE_SOLVER_Z3
Z3 = $(SATDIR)/z3
CXXFLAGS+= -DZ3
Z3_SOLVER_LIBS+= -L$(Z3)/lib -lz3
INCLUDES+= -I $(Z3)/include
endif

ifdef USE_SOLVER_METASMT
CXXFLAGS+= -DMETASMT
METASMT=1
METASMT_SOLVER_LIBS+= -L $(METASMT_PATH)/src/lib -lmetaSMT
endif

ifdef USE_SOLVER_MINISAT
CXXFLAGS+= -DMINISAT
MINISAT=1
METASMT_SOLVER_LIBS+= $(MINISAT_PATH)/core/lib.a  $(MINISAT_PATH)/simp/lib.a
endif

ifdef USE_SOLVER_BOOLECTOR
BOOLECTOR = $(SATDIR)/boolector
CXXFLAGS+= -DBOOLECTOR -I $(BOOLECTOR$)
SOLVER_LIBS+= -L$(BOOLECTOR) -lboolector -L$(SATDIR)/lingeling -llgl
endif

ifdef USE_SOLVER_SWORD
SWORD=1
CXXFLAGS+= -DSWORD
SOLVER_LIBS+= -L$(SWORD_PATH)/lib -lsword
endif

ifdef USE_SOLVER_STP
STP=1
CXXFLAGS+= -DUSESTP
SOLVER_LIBS+= -L$(STP_PATH)/lib -lstp
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
# g++ 4.7 on debian completely fails to link static Z3 lib :|
#FINLINKBIN=g++-4.6
# Compiling static on debian is failing anyway, so don't try to fix it to work.
FINLINKBIN=g++
endif

ifdef LINUXCOMPAT
CC=gcc34
CXX=g++34
FINLINKBIN=g++
FINLINKTAIL+= -Wl,--defsym=memcpy=memmove
STATICLIBCPPPATH:=$(shell g++ $(CXXFLAGS) -print-file-name=libstdc++.a)
LIBS+= $(STATICLIBCPPPATH)

# Link statically against Z3.
ifdef USE_SOLVER_Z3
Z3_SOLVER_LIBS= $(Z3)/lib/libz3.a
endif

else
# If not in compat mode, link against libgomp. Can't do it with g++34 for
# an unknown reason.
PLATLIBS+= -lgomp -pthread -lrt -lpthread
endif

SOLVER_LIBS+=$(Z3_SOLVER_LIBS) $(METASMT_SOLVER_LIBS)

HOSTCC=     gcc
HOSTCXX=    g++
HOSTDEPCXX= g++
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
