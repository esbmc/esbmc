# Default configuration is linux; alternately, uncomment one of the following
# variables. If building on windows, you must have g++ installed for dependancy
# generation, and build in a cygwin environment.

#LINUXCOMPAT=1
#WIN_MSVC=1
WIN_MINGW32=1

# Defaults:

CC=     gcc
CXX=    g++
DEPCXX= g++
LD=     ld
AR=     ar
YACC   = bison -y
YFLAGS = -v
LEX    = flex
PARTLINKLD= ld
PARTLINKOUT= -o
PARTLINKOPTS= -r
FINLINKBIN= $(CXX)
FINLINKOUT= -o
FINLINKTAIL=

ifdef WIN_MINGW32 
CC     = i686-pc-mingw32-gcc
CXX    = i686-pc-mingw32-g++
LD     = i686-pc-mingw32-ld
AR     = i686-pc-mingw32-ar
PARTLINKLD= $(LD)
C2GOTO_INCDIR=C:\\cygwin\\usr\\i686-pc-mingw32\\sys-root\\mingw\\include
WINDOWS= 1
endif

ifdef WIN_MSVC
VCDIR="C:\\Program Files (x86)\\Microsoft Visual Studio 10.0\\VC"
CYGVCDIR="/cygdrive/c/Program Files (x86)/Microsoft Visual Studio 10.0/VC"
VCINCLUDES+=/I$(VCDIR)\\include
VCBINS=$(VCDIR)/bin/amd64
WINESBMCDIR= $(shell cygpath -w $(ESBMCDIR) | sed s/\\\\/\\\\\\\\/g )
WINSDKDIR=C:\\Program\ Files\ \(x86\)\\Microsoft\ SDKs\\Windows\\v7.0A
LD= $(VCBINS)\\link.exe
C2GOTO_INCS=-I$(VCDIR)\\include -I$(WINSDKDIR)\\include
CXX=$(VCBINS)\\cl.exe
CC=$(VCBINS)\\cl.exe
FINLINKBIN= $(CYGVCDIR)/bin/amd64/cl.exe
FINLINKOUT= /Fe $(CXXFLAGS) /link /LIBPATH:$(VCDIR)\\lib\\amd64 /LIBPATH:"C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v7.0A\\lib\\x64\\"
FINLINKTAIL=
endif

# Linux etc.
#you need to install the packages glibc-static and gmp-static to compile statically
OPTS ?= -O3
GCCFLAGS ?= #-Wall -Wextra -Wno-deprecated
#LDFLAGS=

# Work with latest version of gcc
GCCVERSIONGTEQ460 := $(shell expr 4.6.0 \<= `gcc -dumpversion`)
ifeq "$(GCCVERSIONGTEQ460)" "1"
  CXXFLAGS += -fpermissive
endif

# Unless someone else has set OBJDIR, default to `pwd`/.objs
OBJDIR?= .objs
# Danger: evaluating expression creates directory :o
CREATEOBJDIR := $(shell mkdir $(OBJDIR) 2>/dev/null)

# use these for Cygwin:
# OPTS = -O2
# GCCFLAGS = -Wall -mno-cygwin
# LDFLAGS = -mno-cygwin

# SMT and SAT solver locations. Put your own machine-specific absolute paths to
# these in "local.inc", and then don't commit it. Alternately, set SATDIR to
# point at a directory containing such solvers in your environment.
#
# Currently only Z3 is required for building ESBMC.
