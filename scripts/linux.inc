OPTS ?= -O3
PLATLIBS= -lgmp -ldl -lgcc

# Solvers
ifdef USE_SOLVER_Z3
Z3 = $(SATDIR)/z3
CXXFLAGS+= -DZ3
Z3_SOLVER_LIBS+= -L$(Z3)/lib -lz3
INCLUDES+= -I $(Z3)/include
endif

ifdef USE_SOLVER_MINISAT
CXXFLAGS+= -DMINISAT
MINISAT=1
SOLVER_LIBS+= $(MINISAT_PATH)/core/lib.a  $(MINISAT_PATH)/simp/lib.a
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

ifdef USE_SOLVER_MATHSAT
MATHSAT = $(MATHSAT_PATH)
CXXFLAGS+= -DMATHSAT -I $(MATHSAT)/include
SOLVER_LIBS += -L$(MATHSAT)/lib -lmathsat
endif

ifdef USE_SOLVER_CVC
CVC = $(CVC_PATH)
CXXFLAGS+= -DUSECVC -I $(CVC)/include
SOLVER_LIBS += -L$(CVC)/lib -lcvc4
endif

ifdef USE_SOLVER_YICES
YICES = $(YICES_PATH)
CXXFLAGS+= -DYICES -I $(YICES)/include
SOLVER_LIBS += -L$(YICES)/lib -lyices
endif

ifdef USE_EIGEN_LIB
INCLUDES += -I/usr/include/eigen3/
CXXFLAGS+= -DEIGEN_MPL2_ONLY -DEIGEN_LIB
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

# Link statically against Z3.
ifdef USE_SOLVER_Z3
Z3_SOLVER_LIBS= $(Z3)/lib/libz3.a
endif

PLATLIBS+= -lgomp -pthread -lrt -lpthread

SOLVER_LIBS+=$(Z3_SOLVER_LIBS)

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
