all: solvers.o

include ../config.inc
include ../common
include ../local.inc

OBJS= sat/satsolvers.o flattening/flattening.o qbf/qbfsolvers.o \
	prop/propobj.o

DIRS= sat flattening qbf prop

###############################################################################

ifdef BOOLECTOR
OBJS+= boolector/boolector.o
DIRS+= boolector
endif

ifdef Z3
OBJS+= z3/z3.o
DIRS+= z3
endif

ifdef USE_CVC
OBJS+= cvc/cvc.o
DIRS+= cvc
endif

ifdef USE_DPLIB
OBJS+= dplib/dplib.o
DIRS+= dplib
endif

ifdef USE_SMT
OBJS+= smt/smt.o
DIRS+= smt
endif

###############################################################################

$(DIRS):
	$(MAKE) -C $@

$(OBJS): $(DIRS)

.PHONY: $(DIRS)

solvers.o: $(OBJS)
	$(LD) $(LDFLAGS) -r -o solvers.o $(OBJS)

clean:
	-rm solvers.o
	-$(MAKE) -C sat clean
	-$(MAKE) -C flattening clean
	-$(MAKE) -C qbf clean
	-$(MAKE) -C prop clean
	-$(MAKE) -C boolector clean
	-$(MAKE) -C z3 clean
	-$(MAKE) -C cvc clean
	-$(MAKE) -C smt clean
	-$(MAKE) -C dplib clean
