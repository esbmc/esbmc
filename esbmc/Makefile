ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc

SRCS = main.cpp parseoptions.cpp bmc.cpp  \
      document_subgoals.cpp  show_vcc.cpp options.cpp

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

esbmc: $(OBJS)
	echo -n "ESBMC built from " > $(OBJDIR)/buildidobj.s
	echo -n `git rev-parse HEAD` >> $(OBJDIR)/buildidobj.s
	echo -n " " >> $(OBJDIR)/buildidobj.s
	echo -n `date` >> $(OBJDIR)/buildidobj.s
	echo -n " by $(USER)@" >> $(OBJDIR)/buildidobj.s
	echo -n `hostname` >> $(OBJDIR)/buildidobj.s
	# Look for a dirty tree - any files that aren't '??' (i.e., untracked)
	# or config.inc, which might differ for releases.
	-if test -z "`git status -s | grep -v "^??" | grep -v config.inc`"; then echo -n "" >> $(OBJDIR)/buildidobj.s; else echo -n " (dirty tree)" >> $(OBJDIR)/buildidobj.s; fi
	sh -c 'cd $(OBJDIR); $(LD) $(LDFLAGS) -b binary -r ./buildidobj.s -o buildidobj.o'
	cd ..
	rm $(OBJDIR)/buildidobj.s
	$(FINLINKBIN) $(FINLINKOUT)esbmc $(OBJS) $(SOLVER_LIBS) $(OBJDIR)/buildidobj.o $(FINLINKTAIL)

clean:
	-rm -f $(OBJDIR)/* esbmc $(OBJDIR)/.deps
