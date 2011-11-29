ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/common

CXXFLAGS+= -I../../util -I../../

SRCS= cnf.cpp dimacs_cnf.cpp cnf_clause_list.cpp pbs_dimacs_cnf.cpp \
	read_dimacs_cnf.cpp resolution_proof.cpp satcheck.cpp

ifneq ($(CHAFF),)
SRCS+= chaff.cpp
endif

ifneq ($(BOOLEFORCE),)
SRCS+= booleforce.cpp
endif

ifneq ($(MINISAT),)
SRCS+= minisat.cpp
CXXFLAGS+= -DHAVE_MINISAT
endif

ifneq ($(MINISAT2),)
SRCS+= minisat2.cpp
CXXFLAGS+= -DHAVE_MINISAT2
endif

ifneq ($(SMVSAT),)
SRCS+= smvsat.cpp
endif

all: satsolvers.o

satsolvers.o: $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS)

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
	-rm *.o
	-rm *.d

.PHONY: clean
