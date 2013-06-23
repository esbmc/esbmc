all: realall

ifndef METASMT_PATH
$(error You must set METASMT_PATH to build metasmt)
endif

# Munge includes path
INCLUDES+= -I $(METASMT_PATH)/src

ESBMCDIR=$(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJNAME=meta_smt_all.o

SRCS= metasmt_conv.cpp

clean: -rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/.deps

# Can't strictcompile on account of warnings from metaSMT.
#STRICTCOMPILE=1

include $(ESBMCDIR)/common

realall: $(OBJDIR)/.deps $(MAINOBJ)
