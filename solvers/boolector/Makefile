all: realall
ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJNAME=boolector.o

SRCS= boolector_dec.cpp boolector_prop.cpp boolector_conv.cpp boolector_get.cpp

INCLUDES= -I$(BOOLECTOR)/include

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/.deps

include $(ESBMCDIR)/common

realall: $(OBJDIR)/.deps $(MAINOBJ)
