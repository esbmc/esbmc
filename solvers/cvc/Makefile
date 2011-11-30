ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc
MAINOBJ= $(OBJDIR)/cvc.o
all: .deps $(MAINOBJ)

SRCS = cvc_prop.cpp cvc_conv.cpp cvc_dec.cpp

clean:
	-rm -f $(MAINOBJ) $(OBJS) .deps

include $(ESBMCDIR)/common
