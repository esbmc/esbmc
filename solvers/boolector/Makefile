ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/common

CXXFLAGS+= -I../.. -I../../util -I$(BOOLECTOR)/include

SRCS= boolector_dec.cpp boolector_prop.cpp boolector_conv.cpp boolector_get.cpp

all: boolector.o

boolector.o: $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS)

clean:
	-rm *.o
	-rm *.d

.PHONY: clean
