ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/common

CXXFLAGS+= -I../.. -I../../util

SRCS = cvc_prop.cpp cvc_conv.cpp cvc_dec.cpp

all: cvc.o

cvc.o: $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS)

clean:
	-rm *.o
	-rm *.d

.PHONY: clean
