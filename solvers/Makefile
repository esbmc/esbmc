ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc
all: $(OBJDIR)/.depends dummy $(OBJDIR)/solvers.o

OBJS= sat/$(OBJDIR)/satsolvers.o flattening/$(OBJDIR)/flattening.o \
      prop/$(OBJDIR)/propobj.o

DIRS= sat flattening prop

###############################################################################

ifdef BOOLECTOR
OBJS+= boolector/$(OBJDIR)/boolector.o
DIRS+= boolector
endif

ifdef Z3
OBJS+= z3/$(OBJDIR)/z3.o
DIRS+= z3
endif

ifdef USE_CVC
OBJS+= cvc/$(OBJDIR)/cvc.o
DIRS+= cvc
endif

ifdef USE_DPLIB
OBJS+= dplib/$(OBJDIR)/dplib.o
DIRS+= dplib
endif

ifdef USE_SMT
OBJS+= smt/$(OBJDIR)/smt.o
DIRS+= smt
endif

ifdef USE_QBF
OBJS+= qbf/$(OBJDIR)/qbfsolvers.o
DIRS+= qbf
endif

###############################################################################

# Don't rebuild if we're `make clean`ing :|
ifndef DEPENDMAKE
ifndef CLEANRULE
FORCEDIRSMAKE:= $(shell for dir in $(DIRS); do \
                       $(MAKE) -C $$dir;\
                       done)
endif
endif

$(OBJDIR)/solvers.o: $(OBJS)
	$(PARTLINKLD) $(PARTLINKOUT)$@ $^ $(PARTLINKOPTS)

dummy: $(OBJDIRS)

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
	-rm $(OBJDIR)/.depends

include $(ESBMCDIR)/common
