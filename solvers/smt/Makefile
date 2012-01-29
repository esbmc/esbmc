ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc
MAINOBJ= $(OBJDIR)/smt.o
all: $(OBJDIR)/.deps $(MAINOBJ)

SRCS = smt_dec.cpp smt_prop.cpp smt_conv.cpp

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/.deps

include $(ESBMCDIR)/common
