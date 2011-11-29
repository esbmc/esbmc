ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/common

CXXFLAGS+= -I../../ -I../../util

OBJS= qdimacs_cnf.o qbf_quantor.o qbf_skizzo.o qdimacs_core.o qbf_qube.o \
	qbf_qube_core.o

ifneq ($(SQUOLEM2),)
TARGETS+= qbf_squolem.o
CXXFLAGS+= -DHAVE_QBF_CORE
endif

ifneq ($(CUDD),)
TARGETS+= qbf_cudd.o
CXXFLAGS+= -DHAVE_QBF_CORE
endif

qbfsolvers.o: $(TARGETS) $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS)

qbf_squolem.o: qbf_squolem.cpp qbf_squolem_core.cpp
	$(CXX) -c -o qbf_squolem_tmp.o $^ $(CXXFLAGS) -I $(SQUOLEM2)
	$(LD) -r -o $@ qbf_squolem_tmp.o $(LDFLAGS) -L $(SQUOLEM2) -lsquolem2

qbf_cudd.o: qbf_bdd_core.cpp qbf_skizzo_core.cpp
	$(CXX) -c -o qbf_cudd_tmp.o $^ $(CXXFLAGS) -I $(CUDD)/include
	$(LD) -r -o $@ qbf_cudd_tmp.o $(LDFLAGS) -L $(CUDD)/cudd \
		-L $(CUDD)/util -L $(CUDD)/mtr -L $(CUDD)/st -L $(CUDD)/obj \
		-L $(CUDD)/dddmp -L $(CUDD)/epd -lcudd -lmtr -lst -lutil \
		-lepd -ldddmp

clean:
	-rm *.o
	-rm *.d

.PHONY: clean
