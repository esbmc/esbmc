ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc
MAINOBJ= $(OBJDIR)/dplib.o
all: $(OBJDIR)/.deps $(MAINOBJ)

SRCS = dplib_dec.cpp dplib_prop.cpp dplib_conv.cpp

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/.deps

include $(ESBMCDIR)/common
