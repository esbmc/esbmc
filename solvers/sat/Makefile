include ../../config.inc
include ../../common

CPLUSFLAGS+= -I../../util -I../../

ifneq ($(CHAFF),)
TARGETS+= chaff.o
endif

ifneq ($(BOOLEFORCE),)
TARGETS+= booleforce.o
endif

ifneq ($(MINISAT),)
TARGETS+= minisat.o
CPLUSFLAGS+= -DHAVE_MINISAT
endif

ifneq ($(MINISAT2),)
TARGETS+= minisat2.o
CPLUSFLAGS+= -DHAVE_MINISAT2
endif

ifneq ($(SMVSAT),)
TARGETS+= smvsat.o
endif

all: satsolvers.o

satsolvers.o: $(TARGETS)
	$(LD) -r -o $@ $< $(LDFLAGS)

chaff.o: satcheck_zchaff.cpp satcheck_zcore.cpp
	$(CXX) -c -o $@ $< $(CPLUSFLAGS) -lsat -L $(CHAFF) -I $(CHAFF)

booleforce.o: satcheck_booleforce.cpp
	$(CXX) -c -o $@ $< $(CPLUSFLAGS) -lbooleforce -L $(BOOLEFORCE) -I $(BOOLEFORCE)

minisat.o: satcheck_minisat.cpp
	$(CXX) -c -o minisat_tmp.o $< $(CPLUSFLAGS) -I $(MINISAT) -DHAVE_MINISAT
	$(LD) -o $@ minisat_tmp.o -r $(MINISAT)/Solver.o $(MINISAT)/Proof.o \
		$(MINISAT)/File.o $(LDFLAGS)

minisat2.o: satcheck_minisat2.cpp
	$(CXX) -c -o minisat2_tmp.o $< $(CPLUSFLAGS) -I $(MINISAT2)/core \
		-I $(MINISAT2)/mtl -I $(MINISAT2)/simp -DHAVE_MINISAT2
	$(LD) -o $@ minisat2_tmp.o -r $(MINISAT2)/simp/SimpSolver.o \
		$(MINISAT2)/core/Solver.o $(LDFLAGS)

smvsat.o: satcheck_smvsat.cpp
	$(CXX) -c -o $@ $< $(CPLUSFLAGS) -I $(SMVSAT)/include $(SMVSAT)/lib/libsmvsat.a

clean:
	-rm *.o

.PHONY: clean
