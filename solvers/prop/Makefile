ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/common

CXXFLAGS+= -I../.. -I ../../util

OBJS= prop.o prop_conv.o prop_conv_store.o aig_formula.o aig.o aig_prop.o

all: propobj.o

propobj.o: $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS)

clean:
	-rm *.o
	-rm *.d

.PHONY: clean
