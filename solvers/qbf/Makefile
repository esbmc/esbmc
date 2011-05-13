include ../../config.inc
include ../../common

ifneq ($(SQUOLEM2),)
TARGETS+= qbf_squolem.o
endif

ifneq ($(CUDD),)
TARGETS+= qbf_cudd.o
endif

ifeq ($(TARGETS),)
all:
else
all: qbfsolvers.o
endif

qbfsolvers.o: $(TARGETS)
	$(LD) -r -o $@ $< $(LDFLAGS)

qbf_squolem.o: qbf_squolem.cpp qbf_squolem_core.cpp
	$(CXX) -c -o qbf_squolem_tmp.o $< $(CPLUSFLAGS) -I $(SQUOLEM2)
	$(LD) -r -o $@ qbf_squolem_tmp.o $(LDFLAGS) -L $(SQUOLEM2) -lsquolem2

qbf_cudd.o: qbf_bdd_core.cpp qbf_skizzo_core.cpp
	$(CXX) -c -o qbf_cudd_tmp.o $< $(CPLUSFLAGS) -I $(CUDD)/include
	$(LD) -r -o $@ qbf_cudd_tmp.o $(LDFLAGS) -L $(CUDD)/cudd \
		-L $(CUDD)/util -L $(CUDD)/mtr -L $(CUDD)/st -L $(CUDD)/obj \
		-L $(CUDD)/dddmp -L $(CUDD)/epd -lcudd -lmtr -lst -lutil \
		-lepd -ldddmp

clean:
	-rm *.o

.PHONY: clean
