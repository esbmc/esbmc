all: realall

ifdef MINISAT_PATH
INCLUDES+= -I $(MINISAT_PATH)
else
$(error You must set MINISAT_PATH to build minisat)
endif

ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJNAME=minisat.o


SRCS= minisat_conv.cpp

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/.deps

# Minisat headers aren't safe.
#STRICTCOMPILE=1

include $(ESBMCDIR)/common

realall: $(OBJDIR)/.deps $(MAINOBJ)
