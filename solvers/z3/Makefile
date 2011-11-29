ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/common

CXXFLAGS+= -I../.. -I../../util -I$(Z3)/include

SRCS= z3_capi.cpp  z3_conv.cpp  z3_dec.cpp  z3_get.cpp  z3_prop.cpp  z3_util.cpp

all:z3.o

z3.o: $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS)

clean:
	-rm *.o
	-rm *.d

.PHONY: clean
