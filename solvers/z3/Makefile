all: realall
ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJNAME=z3.o

SRCS= z3_conv.cpp

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/.deps

STRICTCOMPILE=1

include $(ESBMCDIR)/common

realall: $(OBJDIR)/.deps $(MAINOBJ)
