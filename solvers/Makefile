ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc

OBJS= prop/$(OBJDIR)/propobj.o smt/$(OBJDIR)/smt.o smtlib/$(OBJDIR)/smtlib.o \
      solve.o

DIRS= prop smt smtlib

###############################################################################

ifdef Z3
OBJS+= z3/$(OBJDIR)/z3.o
DIRS+= z3
endif

ifdef METASMT
OBJS+= metasmt/$(OBJDIR)/meta_smt_all.o
DIRS+= metasmt
endif

###############################################################################

# Make dirs then solvers.o _seperately_ so that changes to file mod
# times are noticed and solvers.o relinked.
all: $(OBJDIR)/.depends
	$(MAKE) dirs
	$(MAKE) $(OBJDIR)/solvers.o

dirs:
	for dir in $(DIRS); do \
	    $(MAKE) -C $$dir; \
	    if test $$? -ne 0; then \
	        exit 1; \
	    fi \
	done

$(OBJDIR)/solvers.o: $(OBJS)
	$(PARTLINKLD) $(PARTLINKOUT)$@ $^ $(PARTLINKOPTS)

clean:
	-rm $(OBJDIR)/solvers.o
	for dir in $(DIRS); do \
	    make -C $$dir clean; \
	done
	-rm $(OBJDIR)/.depends

STRICTCOMPILE=1

include $(ESBMCDIR)/common
