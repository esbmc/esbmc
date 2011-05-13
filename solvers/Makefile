all: solvers.o

include ../config.inc
include ../common

ifneq ($(MINISAT),)
CPLUSFLAGS += -DHAVE_MINISAT
endif

ifneq ($(MINISAT2),)
CPLUSFLAGS += -DHAVE_MINISAT2
endif

QBFOBJ=
ifneq ($(SQUOLEM2),)
QBFOBJ= qbf/qbfsolvers.o
else
ifneq ($(CUDD),)
QBFOBJ= qbf/qbfsolvers.o
endif
endif

ifneq ($(BOOLECTOR),)
BOOLECTOR_INCLUDE=-I $(BOOLECTOR)/include
BOOLECTOR_LIB=$(BOOLECTOR)/lib
endif

ifneq ($(Z3),)
Z3_INCLUDE=-I $(Z3)/include
Z3_LIB=$(Z3)/lib
endif


SRC = \
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

INCLUDES= -I .. $(Z3_INCLUDE) $(BOOLECTOR_INCLUDE) -I ../util

###############################################################################

solvers.o: $(OBJ)
	$(MAKE) -C ./sat
	$(MAKE) -C ./qbf
	$(LD) -r -o solvers.o $(OBJ) -L$(BOOLECTOR_LIB) -lboolector \
	sat/satsolvers.o $(QBFOBJ) $(SQUOLEM2_LIB) $(CUDD_LIB)

clean:
	rm -rf $(OBJ)
