# Default configuration is linux; alternately, uncomment one of the following
# variables. If building on windows, you must have g++ installed for dependancy
# generation, and build in a cygwin environment.

LINUX=1
#WIN_MSVC=1
#WIN_MINGW32=1

# 32 or 64 bits?
TARGET64=1
#TARGET32=1

# Enable linux compatibility (gcc34), not yet implemented

#LINUXCOMPAT=1

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
