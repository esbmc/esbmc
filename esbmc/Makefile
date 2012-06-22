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

ifndef STATICLINK
esbmc: $(OBJS)
	$(ESBMCDIR)/scripts/buildidobj.sh $(OBJDIR)
	$(FINLINKBIN) $(FINLINKOUT)esbmc $(OBJS) $(SOLVER_LIBS) $(OBJDIR)/buildidobj.o $(FINLINKTAIL)

else

# This build rule fetches a libc.a from my website that I've compiled to work on
# iridis, aka linux 2.6.18. The library is checked for my pgp signature; then
# linked up into esbmc to produce a library that /should/ work everywhere.
esbmc: $(OBJS)
	$(ESBMCDIR)/scripts/buildidobj.sh $(OBJDIR)
	$(ESBMCDIR)/scripts/fetchlibc.sh libc${BITWIDTH_STR}.a .libc${BITWIDTH_STR}.a $(ESBMCDIR)/scripts/gpg-keyring
	$(FINLINKBIN) $(FINLINKOUT)$@ $(OBJS) $(SOLVER_LIBS) $(OBJDIR)/buildidobj.o $(FINLINKTAIL) -static -nodefaultlibs `$(CXX) $(CXXFLAGS) -print-file-name=libstdc++.a` -lm -ldl `$(CXX) $(CXXFLAGS) -print-file-name=libgcc.a` `$(CXX) $(CXXFLAGS) -print-file-name=libgcc_eh.a` ./.libc${BITWIDTH_STR}.a

endif

clean:
	-rm -f $(OBJDIR)/* esbmc $(OBJDIR)/.deps
