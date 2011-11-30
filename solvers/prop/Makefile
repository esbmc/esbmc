ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc
MAINOBJ= $(OBJDIR)/propobj.o
all: .deps $(MAINOBJ)

SRCS= prop.cpp prop_conv.cpp prop_conv_store.cpp aig_formula.cpp aig.cpp aig_prop.cpp

clean:
	-rm -f $(MAINOBJ) $(OBJS) .deps

include $(ESBMCDIR)/common
