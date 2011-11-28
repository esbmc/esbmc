SRC = main.cpp parseoptions.cpp bmc.cpp  \
      bv_cbmc.cpp document_subgoals.cpp \
      symex_bmc.cpp show_vcc.cpp \
      counterex_pretty.cpp \
      counterex_pretty_greedy.cpp \
      counterex_pretty_pbs.cpp options.cpp
OBJ = $(SRC:.cpp=.o) \
      ../ansi-c/ansi-c.o \
      ../big-int/bigint.o \
      ../goto-programs/goto-programs.o \
      ../goto-symex/goto-symex.o \
      ../pointer-analysis/pointer-analysis.o \
      ../langapi/langapi.o \
      ../solvers/solvers.o \
      ../util/util.o

INCLUDES= -I .. -I ../util -I$(Z3)/include

LIBS= -lgmp -ldl -lgomp

all: esbmc

include ../config.inc
include ../common
include ../local.inc

ifdef MODULE_CPP
  OBJ += ../cpp/cpp.o
  CXXFLAGS += -DUSE_CPP
endif

ifdef MODULE_SPECC
  OBJ += ../specc/specc.o
  CXXFLAGS += -DUSE_SPECC
endif

ifdef MODULE_PHP
  OBJ += ../php/php.o
  CXXFLAGS += -DUSE_PHP
endif

ifdef BOOLECTOR
CXXFLAGS+= -DBOOLECTOR
LIBS+= -L$(BOOLECTOR)/lib -lboolector
endif

ifdef Z3
CXXFLAGS+= -DZ3
LIBS+= -L$(Z3)/lib -lz3
endif

ifdef MINISAT
CXXFLAGS += -DMINISAT
endif

ifdef USE_CVC
CXXFLAGS+= -DUSE_CVC
endif

ifdef USE_SMT
CXXFLAGS+= USE_SMT
endif

###############################################################################

esbmc: $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o esbmc $(OBJ) $(LIBS)

#cbmc-static: $(OBJ)
#	$(CXX) $(LDFLAGS) -static -static-libgcc -Wl,-Bstatic -o cbmc-static $(OBJ) $(LIBS)

clean:
	rm -f *.o *.d esbmc
