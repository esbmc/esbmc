ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc
MAINOBJ= $(OBJDIR)/dplib.o
all: .deps $(MAINOBJ)

SRCS = dplib_dec.cpp dplib_prop.cpp dplib_conv.cpp

clean:
	-rm -f $(MAINOBJ) $(OBJS).deps

include $(ESBMCDIR)/common
