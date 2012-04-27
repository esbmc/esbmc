ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJ= $(OBJDIR)/propobj.o
all: $(OBJDIR)/.deps $(MAINOBJ)

SRCS= prop_conv.cpp pointer_logic.cpp

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/.deps

include $(ESBMCDIR)/common
