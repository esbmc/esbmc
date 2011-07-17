all: solvers.o

include ../config.inc
include ../common
include ../local.inc

ifdef BOOLECTOR
BOOLBUILD= boolector/boolector.o
endif

ifdef Z3
Z3BUILD= z3/z3.o
endif

###############################################################################

satdir:
	$(MAKE) -C sat

qbfdir:
	$(MAKE) -C qbf

flatteningdir:
	$(MAKE) -C flattening

propdir:
	$(MAKE) -C prop

boolectordir:
	$(MAKE) -C boolector

z3dir:
	$(MAKE) -C z3

cvcdir:
	$(MAKE) -C cvc

smtdir:
	$(MAKE) -C smt

dplibdir:
	$(MAKE) -C dplib

OBJS= sat/satsolvers.o flattening/flattening.o qbf/qbfsolvers.o \
	prop/propobj.o cvc/cvc.o smt/smt.o  dplib/dplib.o \
	$(BOOLBUILD) $(Z3BUILD)

DIRS= satdir flatteningdir qbfdir propdir cvcdir smtdir dplibdir boolectordir \
	z3dir

.PHONY: $(DIRS)

solvers.o: $(DIRS) $(OBJS)
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
