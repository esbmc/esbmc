ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc

SRCS = main.cpp parseoptions.cpp bmc.cpp  \
      bv_cbmc.cpp document_subgoals.cpp \
      symex_bmc.cpp show_vcc.cpp \
      counterex_pretty.cpp \
      counterex_pretty_greedy.cpp \
      counterex_pretty_pbs.cpp options.cpp

# additional objects for esbmc binary
OBJS+=../ansi-c/$(OBJDIR)/ansi-c.o \
      ../big-int/$(OBJDIR)/bigint.o \
      ../goto-programs/$(OBJDIR)/goto-programs.o \
      ../goto-symex/$(OBJDIR)/goto-symex.o \
      ../pointer-analysis/$(OBJDIR)/pointer-analysis.o \
      ../langapi/$(OBJDIR)/langapi.o \
      ../solvers/$(OBJDIR)/solvers.o \
      ../util/$(OBJDIR)/util.o \
      ../cpp/$(OBJDIR)/cpp.o \
      $(OBJDIR)/buildidobj.o

all: $(OBJDIR)/.deps esbmc

ifdef BOOLECTOR
CXXFLAGS+= -DBOOLECTOR
LIBS+= -L$(BOOLECTOR)/lib -lboolector
endif

ifdef Z3
INCLUDES= $(INCCMD)$(Z3)$(PATHSEP)include
CXXFLAGS+= -DZ3
endif

ifdef WIN_MSVC
LIBS+= $(Z3)\\x64\\z3.lib "$(WINSDKDIR)\\lib\\x64\\Kernel32.lib"
endif
ifdef WIN_MINGW32
# Vastly irritating, is that mingw linking against z3.lib causes silent linking
# errors at runtime. So, link directly against the dll.
LIBS+= $(Z3)/$(Z3LIBDIR)/z3.dll
endif
ifdef LINUX
LIBS+= -L $(Z3)/lib -lz3
endif

ifdef MINISAT
CXXFLAGS += -DMINISAT
endif

ifdef USE_CVC
CXXFLAGS+= -DUSE_CVC
endif

ifdef USE_SMT
CXXFLAGS+= USE_SMT
endif

###############################################################################

$(OBJDIR)/buildidobj.o:
	touch $(OBJDIR)/buildidobj.s
	echo -n "ESBMC built from " >> $(OBJDIR)/buildidobj.s
	echo -n `git rev-parse HEAD` >> $(OBJDIR)/buildidobj.s
	echo -n " " >> $(OBJDIR)/buildidobj.s
	echo -n `date` >> $(OBJDIR)/buildidobj.s
	echo -n " by $(USER)@" >> $(OBJDIR)/buildidobj.s
	echo -n `hostname` >> $(OBJDIR)/buildidobj.s
	# Look for a dirty tree - any files that aren't '??' (i.e., untracked)
	# or config.inc/local.inc, which might differ for releases.
	-git status -s | grep -v "^??" | grep -v config.inc | grep -v local.inc | > /dev/null 2>&1
	if test $$? = "0"; then echo -n " (dirty tree)" >> $(OBJDIR)/buildidobj.s; fi
	sh -c 'cd $(OBJDIR); $(LD) $(LDFLAGS) -b binary -r ./buildidobj.s -o buildidobj.o'
	cd ..
	rm $(OBJDIR)/buildidobj.s

include $(ESBMCDIR)/common

esbmc: $(OBJS)
	$(FINLINKBIN) $(FINLINKOUT)esbmc $(OBJS) $(FINLINKTAIL)

clean:
	-rm -f $(OBJDIR)/* esbmc
