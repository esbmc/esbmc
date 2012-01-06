ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc

SRCS = main.cpp parseoptions.cpp bmc.cpp  \
      bv_cbmc.cpp document_subgoals.cpp \
      symex_bmc.cpp show_vcc.cpp \
      counterex_pretty.cpp \
      counterex_pretty_greedy.cpp \
      counterex_pretty_pbs.cpp options.cpp

# Additional include for Z3, as z3_conv.h depends on it.
INCLUDES= -I $(Z3)/include

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

ifndef WINDOWS
LIBS= -lgmp -ldl -lgomp
else
LIBS= wsock32.lib
endif

all: $(OBJDIR)/.deps esbmc

ifdef BOOLECTOR
CXXFLAGS+= -DBOOLECTOR
LIBS+= -L$(BOOLECTOR)/lib -lboolector
endif

ifdef Z3
CXXFLAGS+= -DZ3
LIBS+= $(Z3)\\x64\\z3.lib $(WINSDKDIR)\\lib\\x64\\Kernel32.lib
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

include $(ESBMCDIR)/common

esbmc: $(OBJS)
	$(CXX) $(CXXFLAGS) /OUT:esbmc $(OBJS) $(LIBS) /link /LIBPATH:$(WINSDKDIR)\\lib\\x64 /LIBPATH:$(VCDIR)\\lib\\amd64

#cbmc-static: $(OBJS)
#	$(CXX) -static -static-libgcc -Wl,-Bstatic -o cbmc-static $(OBJS) $(LIBS)

clean:
	-rm -f $(OBJDIR)/* esbmc
