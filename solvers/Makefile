ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc

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

# Make dirs then solvers.o _seperately_ so that changes to file mod
# times are noticed and solvers.o relinked.
all: $(OBJDIR)/.depends
	$(MAKE) dirs
	$(MAKE) $(OBJDIR)/solvers.o

dirs:
	@for dir in $(DIRS); do \
	    $(MAKE) -C $$dir; \
	done

$(OBJDIR)/solvers.o: $(OBJS)
	$(PARTLINKLD) $(PARTLINKOUT)$@ $^ $(PARTLINKOPTS)

clean:
	-rm $(OBJDIR)/solvers.o
	@for dir in $(DIRS); do \
	    make -C $$dir clean; \
	done
	-rm $(OBJDIR)/.depends

include $(ESBMCDIR)/common
