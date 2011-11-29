ESBMCDIR= $(shell pwd)/../..

include ../../config.inc
include ../../common
include ../../local.inc

CXXFLAGS+= -I../.. -I ../../util

OBJS= prop.o prop_conv.o prop_conv_store.o aig_formula.o aig.o aig_prop.o

all: propobj.o

propobj.o: $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS)

clean:
	-rm *.o
	-rm *.d

.PHONY: clean
