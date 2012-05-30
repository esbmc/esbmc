ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc

SRCS = main.cpp parseoptions.cpp bmc.cpp  \
      bv_cbmc.cpp document_subgoals.cpp \
      show_vcc.cpp options.cpp

# additional objects for esbmc binary
OBJS+=../ansi-c/$(OBJDIR)/ansi-c.o \
      ../big-int/$(OBJDIR)/bigint.o \
      ../goto-programs/$(OBJDIR)/goto-programs.o \
      ../goto-symex/$(OBJDIR)/goto-symex.o \
      ../pointer-analysis/$(OBJDIR)/pointer-analysis.o \
      ../langapi/$(OBJDIR)/langapi.o \
      ../solvers/$(OBJDIR)/solvers.o \
      ../util/$(OBJDIR)/util.o \
      ../cpp/$(OBJDIR)/cpp.o

all: $(OBJDIR)/.deps esbmc

###############################################################################

include $(ESBMCDIR)/common

export LD LDFLAGS
esbmc: $(OBJS)
	$(ESBMCDIR)/scripts/buildidobj.sh $(OBJDIR)
	$(FINLINKBIN) $(FINLINKOUT)esbmc $(OBJS) $(SOLVER_LIBS) $(OBJDIR)/buildidobj.o $(FINLINKTAIL)

clean:
	-rm -f $(OBJDIR)/* esbmc $(OBJDIR)/.deps
