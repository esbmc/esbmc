all: realall
ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJNAME=minisat.o

SRCS= minisat_conv.cpp

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/.deps

STRICTCOMPILE=1

include $(ESBMCDIR)/common

realall: $(OBJDIR)/.deps $(MAINOBJ)
