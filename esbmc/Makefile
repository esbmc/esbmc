ESBMCDIR= $(shell pwd)/..

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
      ../util/$(OBJDIR)/util.o

LIBS= -lgmp -ldl -lgomp

all: esbmc

include $(ESBMCDIR)/common

ifdef BOOLECTOR
CXXFLAGS+= -DBOOLECTOR
LIBS+= -L$(BOOLECTOR)/lib -lboolector
endif

ifdef Z3
CXXFLAGS+= -DZ3
LIBS+= -L$(Z3)/lib -lz3
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

esbmc: $(OBJS)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o esbmc $(OBJS) $(LIBS)

#cbmc-static: $(OBJS)
#	$(CXX) $(LDFLAGS) -static -static-libgcc -Wl,-Bstatic -o cbmc-static $(OBJS) $(LIBS)

clean:
	-rm -f $(OBJS) .deps esbmc
