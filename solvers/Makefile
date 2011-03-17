all: solvers.o

include ../config.inc
include ../common

ifneq ($(CHAFF),)
  CHAFF_SRC=sat/satcheck_zchaff.cpp sat/satcheck_zcore.cpp
  CHAFF_INCLUDE=-I ../$(CHAFF)
  CHAFF_LIB=../$(CHAFF)/libsat.a
endif

ifneq ($(BOOLEFORCE),)
  BOOLEFORCE_SRC=sat/satcheck_booleforce.cpp
  BOOLEFORCE_INCLUDE=-I ../$(BOOLEFORCE)
  BOOLEFORCE_LIB=../$(BOOLEFORCE)/libbooleforce.a
endif

ifneq ($(MINISAT),)
  MINISAT_SRC=sat/satcheck_minisat.cpp 
  MINISAT_INCLUDE=-I ../$(MINISAT)
  MINISAT_LIB=../$(MINISAT)/Solver.o ../$(MINISAT)/Proof.o ../$(MINISAT)/File.o
  CPLUSFLAGS += -DHAVE_MINISAT
endif

ifneq ($(MINISAT2),)
  MINISAT2_SRC=sat/satcheck_minisat2.cpp
  MINISAT2_INCLUDE=-I ../$(MINISAT2)/core -I ../$(MINISAT2)/mtl -I ../$(MINISAT2)/simp
  MINISAT2_LIB=../$(MINISAT2)/simp/SimpSolver.o ../$(MINISAT2)/core/Solver.o
  CPLUSFLAGS += -DHAVE_MINISAT2
endif

ifneq ($(SMVSAT),)
  SMVSAT_SRC=sat/satcheck_smvsat.cpp
  SMVSAT_INCLUDE=-I ../$(SMVSAT)/include
  SMVSAT_LIB=../$(SMVSAT)/lib/libsmvsat.a
endif

ifneq ($(SQUOLEM2),)
  SQUOLEM2_SRC=qbf/qbf_squolem.cpp qbf/qbf_squolem_core.cpp
  SQUOLEM2_INCLUDE=-I ../$(SQUOLEM2)
  SQUOLEM2_LIB=-L ../$(SQUOLEM2) -lsquolem2
endif

ifneq ($(CUDD),)
  CUDD_SRC=qbf/qbf_bdd_core.cpp qbf/qbf_skizzo_core.cpp
  CUDD_INCLUDE=-I ../$(CUDD)/include
  CUDD_LIB=-L ../$(CUDD)/cudd -L ../$(CUDD)/util -L ../$(CUDD)/mtr \
           -L ../$(CUDD)/st -L ../$(CUDD)/obj -L ../$(CUDD)/dddmp \
           -L ../$(CUDD)/epd -lobj -lcudd -lmtr -lst -lutil -lepd -ldddmp
endif

ifneq ($(BOOLECTOR),)
BOOLECTOR_INCLUDE=-I ../$(BOOLECTOR)/include
BOOLECTOR_LIB=../$(BOOLECTOR)/lib
endif

ifneq ($(Z3),)
Z3_INCLUDE=-I ../$(Z3)/include
Z3_LIB=../$(Z3)/lib
endif

SRC = $(CHAFF_SRC) $(BOOLEFORCE_SRC) $(MINISAT_SRC) $(MINISAT2_SRC) \
      $(SMVSAT_SRC) $(SQUOLEM2_SRC) $(CUDD_SRC) \
      sat/cnf.cpp sat/dimacs_cnf.cpp sat/cnf_clause_list.cpp \
      sat/pbs_dimacs_cnf.cpp sat/read_dimacs_cnf.cpp \
      sat/resolution_proof.cpp sat/satcheck.cpp \
      qbf/qdimacs_cnf.cpp qbf/qbf_quantor.cpp \
      qbf/qbf_skizzo.cpp qbf/qdimacs_core.cpp qbf/qbf_qube.cpp \
      qbf/qbf_qube_core.cpp \
      prop/prop.cpp prop/prop_conv.cpp prop/prop_conv_store.cpp \
      prop/aig_formula.cpp \
      prop/aig.cpp prop/aig_prop.cpp \
      boolector/boolector_dec.cpp boolector/boolector_prop.cpp boolector/boolector_conv.cpp boolector/boolector_get.cpp \
      z3/z3_dec.cpp z3/z3_prop.cpp z3/z3_conv.cpp z3/z3_get.cpp z3/z3_capi.cpp z3/z3_util.cpp \
      cvc/cvc_prop.cpp cvc/cvc_conv.cpp cvc/cvc_dec.cpp \
      smt/smt_dec.cpp smt/smt_prop.cpp smt/smt_conv.cpp \
      dplib/dplib_conv.cpp dplib/dplib_dec.cpp dplib/dplib_prop.cpp \
      flattening/equality.cpp flattening/arrays.cpp \
      flattening/functions.cpp flattening/sat_minimizer.cpp \
      flattening/boolbv_width.cpp flattening/boolbv.cpp \
      flattening/bv_pointers.cpp flattening/bv_utils.cpp \
      flattening/boolbv_abs.cpp flattening/boolbv_with.cpp \
      flattening/boolbv_typecast.cpp flattening/boolbv_index.cpp \
      flattening/boolbv_member.cpp flattening/boolbv_if.cpp \
      flattening/boolbv_byte_extract.cpp flattening/boolbv_add_sub.cpp \
      flattening/boolbv_mult.cpp flattening/boolbv_constant.cpp \
      flattening/boolbv_extractbit.cpp flattening/boolbv_bv_rel.cpp \
      flattening/boolbv_shift.cpp flattening/boolbv_case.cpp \
      flattening/boolbv_cond.cpp flattening/boolbv_concatenation.cpp \
      flattening/boolbv_div.cpp flattening/boolbv_mod.cpp \
      flattening/boolbv_extractbits.cpp flattening/boolbv_replication.cpp \
      flattening/boolbv_reduction.cpp flattening/boolbv_overflow.cpp \
      flattening/boolbv_get.cpp flattening/boolbv_bitwise.cpp \
      flattening/boolbv_equality.cpp flattening/boolbv_unary_minus.cpp \
      flattening/boolbv_ieee_float_rel.cpp flattening/pointer_logic.cpp \
      flattening/boolbv_quantifier.cpp \
      flattening/boolbv_byte_update.cpp flattening/boolbv_array_of.cpp \
      flattening/boolbv_map.cpp flattening/boolbv_type.cpp \
      flattening/boolbv_union.cpp

OBJ = $(SRC:.cpp=.o)

ifneq ($(SQUOLEM2),)
  CPLUSFLAGS += -DHAVE_QBF_CORE
else
ifneq ($(CUDD),)
  CPLUSFLAGS += -DHAVE_QBF_CORE
endif
endif

ifdef MODULE_FLOATBV
  CPLUSFLAGS += -DHAVE_FLOATBV
  OBJ += ../floatbv/floatbv.o
endif

INCLUDES= -I .. \
 $(CHAFF_INCLUDE) $(BOOLEFORCE_INCLUDE) $(MINISAT_INCLUDE) $(MINISAT2_INCLUDE) \
 $(SMVSAT_INCLUDE) $(SQUOLEM2_INC) $(CUDD_INCLUDE) -I ../util 

###############################################################################

solvers.o: $(OBJ)
	$(LD) -r -o solvers.o $(OBJ) -L$(BOOLECTOR_LIB) -lboolector\
	$(CHAFF_LIB) $(BOOLEFORCE_LIB) $(MINISAT_LIB) $(MINISAT2_LIB) \
        $(SMVSAT_LIB) $(SQUOLEM2_LIB) $(CUDD_LIB)

clean:
	rm -f *.o
	rm -f *.d
