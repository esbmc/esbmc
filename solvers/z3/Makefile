ESBMCDIR= $(shell pwd)/../..
MAINOBJ= z3.o
all:z3.o

SRCS= z3_capi.cpp  z3_conv.cpp  z3_dec.cpp  z3_get.cpp  z3_prop.cpp  z3_util.cpp

INCLUDES= -I$(Z3)/include

clean:
	-rm -f *.o *.d

include $(ESBMCDIR)/common
