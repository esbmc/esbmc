ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJ= $(OBJDIR)/z3.o
all: $(OBJDIR)/.deps $(MAINOBJ)

SRCS= z3_capi.cpp  z3_conv.cpp  z3_get.cpp  z3_prop.cpp  z3_util.cpp

INCLUDES= $(INCCMD)$(Z3)$(PATHSEP)include

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/.deps

include $(ESBMCDIR)/common
