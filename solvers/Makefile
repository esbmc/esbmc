all: solvers.o

include ../config.inc
include ../common
include ../local.inc

ifdef $(BOOLECTOR)
BOOLBUILD= boolector/boolector.o
endif

###############################################################################

sat/satsolvers.o:
	$(MAKE) -C sat

qbf/qbfsolvers.o:
	$(MAKE) -C qbf

flattening/flattening.o:
	$(MAKE) -C flattening

prop/propobj.o:
	$(MAKE) -C prop

boolector/boolector.o:
	$(MAKE) -C boolector

z3/z3.o:
	$(MAKE) -C z3

cvc/cvc.o:
	$(MAKE) -C cvc

smt/smt.o:
	$(MAKE) -C smt

dplib/dplib.o:
	$(MAKE) -C dplib

OBJS= sat/satsolvers.o flattening/flattening.o qbf/qbfsolvers.o \
	prop/propobj.o z3/z3.o cvc/cvc.o smt/smt.o  dplib/dplib.o \
	$(BOOLBUILD)

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
