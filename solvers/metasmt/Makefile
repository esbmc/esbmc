all: realall

ifndef METASMT_PATH
$(error You must set METASMT_PATH to build metasmt)
endif

# Munge includes path
INCLUDES+= -I $(METASMT_PATH)/src

ifdef MINISAT_PATH
INCLUDES+= -I $(MINISAT_PATH) -I $(MINISAT_PATH)/..
endif

ESBMCDIR=$(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJNAME=meta_smt_all.o

SRCS=

ifdef MINISAT
SRCS+= minisat.cpp
endif

ifdef METASMT
  ifdef Z3
    SRCS+= z3.cpp
  endif
endif

ifdef BOOLECTOR
SRCS+= boolector.cpp
endif

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/.deps

# Can't strictcompile on account of warnings from metaSMT.
#STRICTCOMPILE=1

include $(ESBMCDIR)/common

realall: $(OBJDIR)/.deps $(MAINOBJ)
