all: realall
ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJNAME=z3.o

SRCS= z3_capi.cpp  z3_conv.cpp  z3_get.cpp  z3_prop.cpp  z3_util.cpp

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/.deps

include $(ESBMCDIR)/common

realall: $(OBJDIR)/.deps $(MAINOBJ)
