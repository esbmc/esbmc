ESBMCDIR= $(shell pwd)/../..

include ../../config.inc
include ../../common
include ../../local.inc

CXXFLAGS+= -I../.. -I../../util

OBJS = dplib_dec.o dplib_prop.o dplib_conv.o

all: dplib.o

dplib.o: $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS)

clean:
	-rm *.o
	-rm *.d

.PHONY: clean

