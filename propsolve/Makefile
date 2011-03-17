all: propsolve.o

include ../config.inc

ifneq ($(CHAFF),)
CHAFF_SRC=satcheck_zchaff.cpp satcheck_zcore.cpp
CHAFF_INCLUDE=-I ../$(CHAFF)
CHAFF_LIB=../$(CHAFF)/libsat.a
endif

ifneq ($(BOOLEFORCE),)
BOOLEFORCE_SRC=satcheck_booleforce.cpp
BOOLEFORCE_INCLUDE=-I ../$(BOOLEFORCE)
BOOLEFORCE_LIB=../$(BOOLEFORCE)/libbooleforce.a
endif

ifneq ($(MINISAT),)
MINISAT_SRC=satcheck_minisat.cpp 
MINISAT_INCLUDE=-I ../$(MINISAT)
MINISAT_LIB=../$(MINISAT)/Solver.o ../$(MINISAT)/Proof.o ../$(MINISAT)/File.o
CPLUSFLAGS += -DHAVE_MINISAT
endif

ifneq ($(MINISAT2),)
MINISAT2_SRC=satcheck_minisat2.cpp
MINISAT2_INCLUDE=-I ../$(MINISAT2)/core -I ../$(MINISAT2)/mtl -I ../$(MINISAT2)/simp
MINISAT2_LIB=../$(MINISAT2)/simp/SimpSolver.o ../$(MINISAT2)/core/Solver.o
CPLUSFLAGS += -DHAVE_MINISAT2
endif

ifneq ($(SMVSAT),)
SMVSAT_SRC=satcheck_smvsat.cpp
SMVSAT_INCLUDE=-I ../$(SMVSAT)/include
SMVSAT_LIB=../$(SMVSAT)/lib/libsmvsat.a
endif

ifneq ($(SQUOLEM2),)
SQUOLEM2_SRC=qbf_squolem.cpp qbf_squolem_core.cpp
SQUOLEM2_INCLUDE=-I ../$(SQUOLEM2)
SQUOLEM2_LIB=-L ../$(SQUOLEM2) -lsquolem2
endif

ifneq ($(CUDD),)
CUDD_SRC=qbf_bdd_core.cpp qbf_skizzo_core.cpp
CUDD_INCLUDE=-I ../$(CUDD)/include
CUDD_LIB=-L ../$(CUDD)/cudd -L ../$(CUDD)/util -L ../$(CUDD)/mtr \
         -L ../$(CUDD)/st -L ../$(CUDD)/obj -L ../$(CUDD)/dddmp \
         -L ../$(CUDD)/epd -lobj -lcudd -lmtr -lst -lutil -lepd -ldddmp
endif

SRC = $(CHAFF_SRC) $(BOOLEFORCE_SRC) $(MINISAT_SRC) $(MINISAT2_SRC) $(SMVSAT_SRC) \
      $(SQUOLEM2_SRC) $(CUDD_SRC) \
      cnf.cpp dimacs_cnf.cpp cnf_clause_list.cpp aig_formula.cpp \
      pbs_dimacs_cnf.cpp qdimacs_cnf.cpp cvc_prop.cpp qbf_quantor.cpp \
      aig.cpp aig_prop.cpp resolution_proof.cpp sat_minimizer.cpp \
      smt_prop.cpp read_dimacs_cnf.cpp qbf_skizzo.cpp dplib_prop.cpp \
      qdimacs_core.cpp qbf_qube.cpp qbf_qube_core.cpp

OBJ = $(SRC:.cpp=.o)

include ../common

ifneq ($(SQUOLEM2),)
CPLUSFLAGS += -DHAVE_QBF_CORE
else
ifneq ($(CUDD),)
CPLUSFLAGS += -DHAVE_QBF_CORE
endif
endif

INCLUDES= -I .. \
 $(CHAFF_INCLUDE) $(BOOLEFORCE_INCLUDE) $(MINISAT_INCLUDE) $(MINISAT2_INCLUDE) \
 $(SMVSAT_INCLUDE) $(SQUOLEM2_INC) $(CUDD_INCLUDE) -I ../util 

###############################################################################

propsolve.o: $(OBJ)
	$(LD) -r -o propsolve.o $(OBJ) \
	$(CHAFF_LIB) $(BOOLEFORCE_LIB) $(MINISAT_LIB) $(MINISAT2_LIB) \
        $(SMVSAT_LIB) $(SQUOLEM2_LIB) $(CUDD_LIB)

clean:
	rm -f *.o
