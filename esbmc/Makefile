SRC = main.cpp parseoptions.cpp bmc.cpp smt.cpp z3.cpp boolector.cpp \
      bv_cbmc.cpp document_subgoals.cpp dimacs.cpp \
      symex_bmc.cpp cvc.cpp show_vcc.cpp \
      counterexample_beautification.cpp \
      counterexample_beautification_greedy.cpp \
      counterexample_beautification_pbs.cpp
OBJ = $(SRC:.cpp=.o) \
      ../ansi-c/ansi-c.o \
      ../big-int/bigint.o \
      ../langapi/csp_dummy.o \
      ../goto-programs/goto-programs.o \
      ../goto-symex/goto-symex.o \
      ../pointer-analysis/pointer-analysis.o \
      ../langapi/intrep_dummy.o \
      ../langapi/langapi.o \
      ../solvers/solvers.o \
      ../langapi/pvs_dummy.o \
      ../langapi/smvlang_dummy.o \
      ../langapi/promela_dummy.o \
      ../util/util.o \
      ../langapi/verilog_dummy.o \
      ../langapi/vhdl_dummy.o \
      ../langapi/smtlang_dummy.o \
      ../langapi/netlist_dummy.o \
      ../langapi/pascal_dummy.o \
      ../langapi/bplang_dummy.o \
      ../langapi/simplifylang_dummy.o \
      ../langapi/csharp_dummy.o \
      ../cvclang/cvclang.o \
      ../langapi/nsf_dummy.o \
      ../langapi/mdllang_dummy.o

INCLUDES= -I .. -I ../util -I$(Z3)/include

LIBS=-L$(BOOLECTOR)/lib -lboolector -L $(Z3)/lib -lz3 -lgmp -ldl

all: esbmc

include ../config.inc
include ../common

ifdef MODULE_BV_REFINEMENT
  OBJ += ../bv_refinement/bv_refinement.o
  CPLUSFLAGS += -DHAVE_BV_REFINEMENT
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

###############################################################################

esbmc: $(OBJ)
	$(CPLUS) $(LINKFLAGS) -o esbmc $(OBJ) $(LIBS)

#cbmc-static: $(OBJ)
#	$(CPLUS) $(LINKFLAGS) -static -static-libgcc -Wl,-Bstatic -o cbmc-static $(OBJ) $(LIBS)

clean:
	rm -f *.o *.d esbmc
