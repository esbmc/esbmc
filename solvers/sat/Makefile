ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc
MAINOBJ= $(OBJDIR)/satsolvers.o
all: $(OBJDIR)/.deps $(MAINOBJ)

SRCS= cnf.cpp cnf_clause_list.cpp \
	resolution_proof.cpp satcheck.cpp

ifdef MINISAT
OBJS+= $(OBJDIR)/minisat.o
CXXFLAGS+= -DHAVE_MINISAT
endif

ifdef MINISAT2
OBJS+= $(OBJDIR)/minisat2.o
CXXFLAGS+= -DHAVE_MINISAT2
endif

$(OBJDIR)/minisat.o: satcheck_minisat.cpp
	$(CXX) -c -o $(OBJDIR)/minisat_tmp.o $^ $(CXXFLAGS) -I $(MINISAT) -DHAVE_MINISAT
	$(LD) -o $@ $(OBJDIR)/minisat_tmp.o -r $(MINISAT)/Solver.o $(MINISAT)/Proof.o \
		$(MINISAT)/File.o $(LDFLAGS)

$(OBJDIR)/minisat2.o: satcheck_minisat2.cpp
	$(CXX) -c -o $(OBJDIR)/minisat2_tmp.o $^ $(CXXFLAGS) -I $(MINISAT2)/core \
		-I $(MINISAT2)/mtl -I $(MINISAT2)/simp -DHAVE_MINISAT2
	$(LD) -o $@ $(OBJDIR)/minisat2_tmp.o -r $(MINISAT2)/simp/SimpSolver.o \
		$(MINISAT2)/core/Solver.o $(LDFLAGS)

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/minisat_tmp.o $(OBJDIR)/.deps

include $(ESBMCDIR)/common
