# Default configuration is linux; alternately, uncomment one of the following
# variables. If building on windows, you must have g++ installed for dependancy
# generation, and build in a cygwin environment.

#LINUX=1
#WIN_MSVC=1
WIN_MINGW32=1

# Enable linux compatibility (gcc34), not yet implemented

#LINUXCOMPAT=1

# Detect cygwin

AMONCYGWIN= $(shell uname -a | grep CYGWIN; echo $$?)
ifeq ($(AMONCYGWIN), 0)
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
PLATLIBS= -lgmp -ldl -lgomp
PARTLINKLD= ld
PARTLINKOUT= -o
PARTLINKOPTS= -r
FINLINKBIN= $(CXX)
FINLINKOUT= -o
FINLINKTAIL= $(LDFLAGS) $(PLATLIBS) $(LIBS)
INCCMD=-I
PATHSEP=/

ifdef LINUX
CXXFLAGS+= -Wno-deprecated
endif

ifdef WIN_MINGW32 
# mingw doesn't seem to have pkg-config.
MINGWROOTDIR= /usr/i686-pc-mingw32/sys-root/mingw
CC     = i686-pc-mingw32-gcc
CXX    = i686-pc-mingw32-g++
LD     = i686-pc-mingw32-ld
AR     = i686-pc-mingw32-ar
CFLAGS+= -I$(MINGWROOTDIR)/include -DWINVER=0x0501
CXXFLAGS+= -I$(MINGWROOTDIR)/include -DWINVER=0x0501 -Wno-deprecated
LDFLAGS+= -lkernel32 -L $(MINGWROOTDIR)/lib
PARTLINKLD= $(LD)
BLOBGENLD= $(LD)
ifdef ONCYGWIN
C2GOTO_INCS=-I$(shell cygpath -w $(MINGWROOTDIR)/include)
else
C2GOTO_INCS=-I$(MINGWROOTDIR)/include -D_MINGW
endif
WINDOWS= 1
MINGWRUNTIME= mingw_runtime_bodge
MINGWRUNOBJS=libstdc++-6.dll libgcc_s_sjlj-1.dll
PLATLIBS=-lws2_32 -lkernel32
FINLINKTAIL= $(LDFLAGS) $(LIBS) $(PLATLIBS)
endif

ifdef WIN_MSVC

WINSDKDIR=$(shell reg query HKLM\\SOFTWARE\\Microsoft\\Microsoft\ SDKs\\Windows /v CurrentInstallFolder | grep REG_SZ | sed "s\# *CurrentInstallFolder *REG_SZ *\(.*\)\#\1\#" | sed s/\\\\/\\\\\\\\/g)

VCDIR=$(shell reg query HKLM\\SOFTWARE\\Microsoft\\VisualStudio\\10.0\\InstalledProducts\\Instrumentation /v Location | grep REG_SZ | sed "s\# *Instrumentation *REG_SZ *\(.*\)\#\1\#" | sed s/\\\\/\\\\\\\\/g)

CYGVCDIR=$(shell cygpath -u $(VCDIR))
VCINCLUDES+=/I$(VCDIR)\\include
VCBINS=$(VCDIR)/bin/amd64
WINESBMCDIR= $(shell cygpath -w $(ESBMCDIR) | sed s/\\\\/\\\\\\\\/g )
LD= $(VCBINS)\\link.exe
C2GOTO_INCS=-I$(VCDIR)\\include -I$(WINSDKDIR)\\include -D_MSVC
CXX=$(VCBINS)\\cl.exe
CC=$(VCBINS)\\cl.exe
FINLINKBIN= $(CYGVCDIR)/bin/amd64/cl.exe
FINLINKOUT= /Fe
BLOBGENLD=x86_64-w64-mingw32-ld
INCCMD=/I
PATHSEP=\\
CXXFLAGS+=/I$(VCDIR)\\include
PLATLIBS="C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v7.0A\\lib\\x64\\wsock32.lib"
FINLINKTAIL= $(CXXFLAGS) /link /LIBPATH:$(VCDIR)\\lib\\amd64 /LIBPATH:"C:\\Program Files (x86)\\Microsoft SDKs\\Windows\\v7.0A\\lib\\x64\\" $(PLATLIBS) $(LIBS)
PARTLINKLD= $(VCBINS)\\lib.exe
PARTLINKOUT= /OUT:
PARTLINKOPTS= 
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
