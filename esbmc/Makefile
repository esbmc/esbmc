SRC = main.cpp parseoptions.cpp bmc.cpp smt.cpp z3.cpp \
      bv_cbmc.cpp document_subgoals.cpp dimacs.cpp \
      symex_bmc.cpp cvc.cpp show_vcc.cpp \
      counterexample_beautification.cpp \
      counterexample_beautification_greedy.cpp \
      counterexample_beautification_pbs.cpp
OBJ = $(SRC:.cpp=.o) \
      ../ansi-c/ansi-c.o \
      ../big-int/bigint.o \
      ../goto-programs/goto-programs.o \
      ../goto-symex/goto-symex.o \
      ../pointer-analysis/pointer-analysis.o \
      ../langapi/langapi.o \
      ../solvers/solvers.o \
      ../util/util.o \
      ../cvclang/cvclang.o

INCLUDES= -I .. -I ../util -I$(Z3)/include

LIBS= -L $(Z3)/lib -lz3 -lgmp -ldl

all: esbmc

include ../config.inc
include ../common
include ../local.inc

ifdef MODULE_BV_REFINEMENT
  OBJ += ../bv_refinement/bv_refinement.o
  CXXFLAGS += -DHAVE_BV_REFINEMENT
endif

ifdef MODULE_CPP
  OBJ += ../cpp/cpp.o
else
  OBJ += ../langapi/cpp_dummy.o
endif

ifdef MODULE_SPECC
  OBJ += ../specc/specc.o
else
  OBJ += ../langapi/specc_dummy.o
endif

ifdef MODULE_PHP
  OBJ += ../php/php.o
else
  OBJ += ../langapi/php_dummy.o
endif

ifdef BOOLECTOR
CXXFLAGS+= -DBOOLECTOR
SRC+= boolector.cpp
LIBS+= -L$(BOOLECTOR)/lib -lboolector
endif

###############################################################################

esbmc: $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o esbmc $(OBJ) $(LIBS)

#cbmc-static: $(OBJ)
#	$(CXX) $(LDFLAGS) -static -static-libgcc -Wl,-Bstatic -o cbmc-static $(OBJ) $(LIBS)

clean:
	rm -f *.o *.d esbmc
