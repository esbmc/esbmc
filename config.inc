# Default configuration is linux; alternately, uncomment one of the following
# variables. If building on windows, you must have g++ installed for dependancy
# generation, and build in a cygwin environment.

ifndef EXTERN_ESBMC_CONFIG # Don't overwrite configuration given by environment

LINUX=1
#WIN_MSVC=1
#WIN_MINGW32=1

# 32 or 64 bits? If left unset, default to whatever your host machine is.
#TARGET64=1
#TARGET32=1

# Static? (Linux only)
#STATICLINK=1

# Enable linux compatibility (gcc34), not yet implemented

#LINUXCOMPAT=1

endif #EXTERN_ESBMC_CONFIG

# Might need to guess what bitwidth target to build.
ifndef TARGET64
ifndef TARGET32
THEMACHINE:=$(uname -m)
ifeq ($(THEMACHINE),x86_64)
  TARGET64=1
else
  TARGET32=1
endif
endif
endif

# Detect cygwin

AMONCYGWIN:= $(shell uname -a | grep -o CYGWIN)
ifeq "$(AMONCYGWIN)" "CYGWIN"
ONCYGWIN=1
endif

# Defaults:

CC=     gcc
CXX=    g++
DEPCXX= g++
LD=     ld
AR=     ar
YACC   = bison -y
YFLAGS = -v
LEX    = flex
BLOBGENLD= ld
PARTLINKLD= ld
PARTLINKOUT= -o
PARTLINKOPTS= -r $(LDFLAGS)
FINLINKBIN= $(CXX)
FINLINKOUT= -o
FINLINKTAIL= $(CXXFLAGS) $(LIBS) $(PLATLIBS)
INCCMD=-I
PATHSEP=/

# If on a 64 bit machine, build C libraries as expected. If on 32 bits, build
# everything 32 bits. This is because you can't get 64 bit headers (easily) on
# 32 bit machines, wheras you can vice versa.
FULLUNAME:=$(shell uname -a)
WEARE64BIT:=$(shell uname -a | egrep "(CYGWIN.*WOW64|x86_64)")
ifneq ($(WEARE64BIT),"")
CLIB64FLAG=--64
CLIB32FLAG=--32
else
CLIB64FLAG=--32
CLIB32FLAG=--32
endif

ifdef LINUX
include $(ESBMCDIR)/scripts/linux.inc
endif

ifdef WIN_MINGW32 
include $(ESBMCDIR)/scripts/mingw.inc
endif

ifdef WIN_MSVC
include $(ESBMCDIR)/scripts/msvc.inc
endif

# Unless someone else has set OBJDIR, default to `pwd`/.objs
OBJDIR?= .objs
# Danger: evaluating expression creates directory :o
CREATEOBJDIR := $(shell mkdir $(OBJDIR) 2>/dev/null)
