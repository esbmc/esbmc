ESBMCDIR= $(shell pwd)/../..
MAINOBJ= qbfsolvers.o
all: $(MAINOBJ)

SRCS= qdimacs_cnf.cpp qbf_quantor.cpp qbf_skizzo.cpp qdimacs_core.cpp \
      qbf_qube.cpp qbf_qube_core.cpp

ifneq ($(SQUOLEM2),)
SRCS+= qbf_squolem.cpp
CXXFLAGS+= -DHAVE_QBF_CORE
endif

ifneq ($(CUDD),)
SRCS+= qbf_cudd.cpp
CXXFLAGS+= -DHAVE_QBF_CORE
endif

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
	-rm -f *.o *.d

include $(ESBMCDIR)/common
