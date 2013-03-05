ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJ= $(OBJDIR)/smtlib.o
all: $(OBJDIR)/.deps $(MAINOBJ)

SRCS= smtlib_conv.cpp

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/.deps


STRICTCOMPILE=1

include $(ESBMCDIR)/common
