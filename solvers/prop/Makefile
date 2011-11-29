ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/common

CXXFLAGS+= -I../.. -I ../../util

SRCS= prop.cpp prop_conv.cpp prop_conv_store.cpp aig_formula.cpp aig.cpp aig_prop.cpp

all: propobj.o

propobj.o: $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS)

clean:
	-rm *.o
	-rm *.d

.PHONY: clean
