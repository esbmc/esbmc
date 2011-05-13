all: solvers.o

include ../config.inc
include ../common

ifneq ($(MINISAT),)
CPLUSFLAGS += -DHAVE_MINISAT
endif

ifneq ($(MINISAT2),)
CPLUSFLAGS += -DHAVE_MINISAT2
endif

ifneq ($(Z3),)
Z3_INCLUDE=-I $(Z3)/include
Z3_LIB=$(Z3)/lib
endif


SRC = \
      z3/z3_dec.cpp z3/z3_prop.cpp z3/z3_conv.cpp z3/z3_get.cpp z3/z3_capi.cpp z3/z3_util.cpp \
      cvc/cvc_prop.cpp cvc/cvc_conv.cpp cvc/cvc_dec.cpp \
      smt/smt_dec.cpp smt/smt_prop.cpp smt/smt_conv.cpp \
      dplib/dplib_conv.cpp dplib/dplib_dec.cpp dplib/dplib_prop.cpp

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

INCLUDES= -I .. $(Z3_INCLUDE) -I ../util

###############################################################################

solvers.o: $(OBJ)
	$(MAKE) -C ./sat
	$(MAKE) -C ./qbf
	$(MAKE) -C ./flattening
	$(MAKE) -C ./prop
	$(MAKE) -C ./boolector
	$(LD) -r -o solvers.o $(OBJ) \
	sat/satsolvers.o flattening/flattening.o qbf/qbfsolvers.o \
	prop/propobj.o boolector/boolector.o $(SQUOLEM2_LIB) $(CUDD_LIB)

clean:
	rm -rf $(OBJ)
