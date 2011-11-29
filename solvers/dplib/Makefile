ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/common

CXXFLAGS+= -I../.. -I../../util

SRCS = dplib_dec.cpp dplib_prop.cpp dplib_conv.cpp

all: dplib.o

dplib.o: $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS)

clean:
	-rm *.o
	-rm *.d

.PHONY: clean

