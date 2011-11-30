ESBMCDIR= $(shell pwd)/../..
include $(ESBMDIR)/config.inc
include $(ESBMDIR)/local.inc
MAINOBJ= $(OBJDIR)/dplib.o
all: $(MAINOBJ)

SRCS = dplib_dec.cpp dplib_prop.cpp dplib_conv.cpp

clean:
	-rm -f $(MAINOBJ) $(OBJS).deps

include $(ESBMCDIR)/common
