ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc

OBJS= sat/$(OBJDIR)/satsolvers.o flattening/$(OBJDIR)/flattening.o \
      prop/$(OBJDIR)/propobj.o

DIRS= sat flattening prop

###############################################################################

ifdef Z3
OBJS+= z3/$(OBJDIR)/z3.o
DIRS+= z3
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
