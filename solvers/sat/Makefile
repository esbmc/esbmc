ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc
MAINOBJ= $(OBJDIR)/satsolvers.o
all: .deps $(MAINOBJ)

SRCS= cnf.cpp dimacs_cnf.cpp cnf_clause_list.cpp pbs_dimacs_cnf.cpp \
	read_dimacs_cnf.cpp resolution_proof.cpp satcheck.cpp

ifdef CHAFF
OBJS+= $(OBJDIR)/chaff.o
endif

ifdef BOOLEFORCE
OBJS+= $(OBJDIR)/booleforce.o
endif

ifdef MINISAT
OBJS+= $(OBJDIR)/minisat.o
CXXFLAGS+= -DHAVE_MINISAT
endif

ifdef MINISAT2
OBJS+= $(OBJDIR)/minisat2.o
CXXFLAGS+= -DHAVE_MINISAT2
endif

ifdef SMVSAT
OBJS+= $(OBJDIR)/smvsat.o
endif

$(OBJDIR)/chaff.o: satcheck_zchaff.cpp satcheck_zcore.cpp
	$(CXX) -c -o $@ $^ $(CXXFLAGS) -lsat -L $(CHAFF) -I $(CHAFF)

$(OBJDIR)/booleforce.o: satcheck_booleforce.cpp
	$(CXX) -c -o $@ $^ $(CXXFLAGS) -lbooleforce -L $(BOOLEFORCE) -I $(BOOLEFORCE)

$(OBJDIR)/minisat.o: satcheck_minisat.cpp
	$(CXX) -c -o $(OBJDIR)/minisat_tmp.o $^ $(CXXFLAGS) -I $(MINISAT) -DHAVE_MINISAT
	$(LD) -o $@ $(OBJDIR)/minisat_tmp.o -r $(MINISAT)/Solver.o $(MINISAT)/Proof.o \
		$(MINISAT)/File.o $(LDFLAGS)

$(OBJDIR)/minisat2.o: satcheck_minisat2.cpp
	$(CXX) -c -o $(OBJDIR)/minisat2_tmp.o $^ $(CXXFLAGS) -I $(MINISAT2)/core \
		-I $(MINISAT2)/mtl -I $(MINISAT2)/simp -DHAVE_MINISAT2
	$(LD) -o $@ $(OBJDIR)/minisat2_tmp.o -r $(MINISAT2)/simp/SimpSolver.o \
		$(MINISAT2)/core/Solver.o $(LDFLAGS)

$(OBJDIR)/smvsat.o: satcheck_smvsat.cpp
	$(CXX) -c -o $@ $^ $(CXXFLAGS) -I $(SMVSAT)/include $(SMVSAT)/lib/libsmvsat.a

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/minisat_tmp.o .deps

include $(ESBMCDIR)/common
