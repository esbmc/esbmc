ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc
MAINOBJ= boolector.o
all: $(MAINOBJ)

SRCS= boolector_dec.cpp boolector_prop.cpp boolector_conv.cpp boolector_get.cpp

INCLUDES= -I$(BOOLECTOR)/include

clean:
	-rm -f *.o *.d

include $(ESBMCDIR)/common

