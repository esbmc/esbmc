all: solvers.o

include ../config.inc
include ../common
include ../local.inc

OBJS= sat/satsolvers.o flattening/flattening.o qbf/qbfsolvers.o \
	prop/propobj.o smt/smt.o  dplib/dplib.o

DIRS= sat flattening qbf prop smt dplib

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
