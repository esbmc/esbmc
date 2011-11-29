ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc
MAINOBJ= satsolvers.o
all: $(MAINOBJ)

SRCS= cnf.cpp dimacs_cnf.cpp cnf_clause_list.cpp pbs_dimacs_cnf.cpp \
	read_dimacs_cnf.cpp resolution_proof.cpp satcheck.cpp

ifdef CHAFF
OBJS+= chaff.o
endif

ifdef BOOLEFORCE
OBJS+= booleforce.o
endif

ifdef MINISAT
OBJS+= minisat.o
CXXFLAGS+= -DHAVE_MINISAT
endif

ifdef MINISAT2
OBJS+= minisat2.cpp
CXXFLAGS+= -DHAVE_MINISAT2
endif

ifdef SMVSAT
OBJS+= smvsat.cpp
endif

chaff.o: satcheck_zchaff.cpp satcheck_zcore.cpp
	$(CXX) -c -o $@ $^ $(CXXFLAGS) -lsat -L $(CHAFF) -I $(CHAFF)

booleforce.o: satcheck_booleforce.cpp
	$(CXX) -c -o $@ $^ $(CXXFLAGS) -lbooleforce -L $(BOOLEFORCE) -I $(BOOLEFORCE)

minisat.o: satcheck_minisat.cpp
	$(CXX) -c -o minisat_tmp.o $^ $(CXXFLAGS) -I $(MINISAT) -DHAVE_MINISAT
	$(LD) -o $@ minisat_tmp.o -r $(MINISAT)/Solver.o $(MINISAT)/Proof.o \
		$(MINISAT)/File.o $(LDFLAGS)

minisat2.o: satcheck_minisat2.cpp
	$(CXX) -c -o minisat2_tmp.o $^ $(CXXFLAGS) -I $(MINISAT2)/core \
		-I $(MINISAT2)/mtl -I $(MINISAT2)/simp -DHAVE_MINISAT2
	$(LD) -o $@ minisat2_tmp.o -r $(MINISAT2)/simp/SimpSolver.o \
		$(MINISAT2)/core/Solver.o $(LDFLAGS)

smvsat.o: satcheck_smvsat.cpp
	$(CXX) -c -o $@ $^ $(CXXFLAGS) -I $(SMVSAT)/include $(SMVSAT)/lib/libsmvsat.a

clean:
	-rm -f *.o *.d

include $(ESBMCDIR)/common
