ESBMCDIR= $(shell pwd)/../..

include ../../config.inc
include ../../common
include ../../local.inc

CXXFLAGS+= -I../.. -I../../util -I$(Z3)/include

OBJS= z3_capi.o  z3_conv.o  z3_dec.o  z3_get.o  z3_prop.o  z3_util.o

all:z3.o

z3.o: $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS)

clean:
	-rm *.o
	-rm *.d

.PHONY: clean
