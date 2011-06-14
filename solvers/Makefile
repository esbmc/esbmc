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

sat/satsolvers.o: satdir

satdir:
	$(MAKE) -C sat

qbf/qbfsolvers.o: qbfdir

qbfdir:
	$(MAKE) -C qbf

flattening/flattening.o: flatteningdir

flatteningdir:
	$(MAKE) -C flattening

prop/propobj.o: propdir

propdir:
	$(MAKE) -C prop

boolector/boolector.o: boolectordir

boolectordir:
	$(MAKE) -C boolector

z3/z3.o: z3dir

z3dir:
	$(MAKE) -C z3

cvc/cvc.o: cvcdir

cvcdir:
	$(MAKE) -C cvc

smt/smt.o: smtdir

smtdir:
	$(MAKE) -C smt

dplib/dplib.o: dplibdir

dplibdir:
	$(MAKE) -C dplib

OBJS= sat/satsolvers.o flattening/flattening.o qbf/qbfsolvers.o \
	prop/propobj.o cvc/cvc.o smt/smt.o  dplib/dplib.o \
	$(BOOLBUILD) $(Z3BUILD)

DIRS= satdir flatteningdir qbfdir propdir cvcdir smtdir dplibdir boolectordir \
	z3dir

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
