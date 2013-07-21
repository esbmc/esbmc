all: realall
ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJNAME=smt.o

SRCS= smt_conv.cpp smt_tuple.cpp smt_overflow.cpp smt_byteops.cpp \
      smt_memspace.cpp smt_casts.cpp array_conv.cpp

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/.deps


STRICTCOMPILE=1

include $(ESBMCDIR)/common

realall: $(OBJDIR)/.deps $(MAINOBJ)
