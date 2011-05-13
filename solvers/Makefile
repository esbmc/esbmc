all: solvers.o

include ../config.inc
include ../common

ifneq ($(MINISAT),)
CPLUSFLAGS += -DHAVE_MINISAT
endif

ifneq ($(MINISAT2),)
CPLUSFLAGS += -DHAVE_MINISAT2
endif

SRC = \
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

INCLUDES= -I .. -I ../util

###############################################################################

solvers.o: $(OBJ)
	$(MAKE) -C ./sat
	$(MAKE) -C ./qbf
	$(MAKE) -C ./flattening
	$(MAKE) -C ./prop
	$(MAKE) -C ./boolector
	$(MAKE) -C ./z3
	$(MAKE) -C ./cvc
	$(MAKE) -C ./smt
	$(LD) -r -o solvers.o $(OBJ) \
	sat/satsolvers.o flattening/flattening.o qbf/qbfsolvers.o \
	prop/propobj.o boolector/boolector.o z3/z3.o cvc/cvc.o smt/smt.o \
	$(SQUOLEM2_LIB) $(CUDD_LIB)

clean:
	rm -rf $(OBJ)
