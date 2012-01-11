ifdef TARGET64
MINGWID=x86_64-w64-mingw32
Z3LIBDIR=x64
else
MINGWID=i686-pc-mingw32
Z3LIBDIR=lib
endif

# mingw doesn't seem to have pkg-config.
MINGWROOTDIR= /usr/$(MINGWID)/sys-root/mingw
CC     = $(MINGWID)-gcc
CXX    = $(MINGWID)-g++
LD     = $(MINGWID)-ld
AR     = $(MINGWID)-ar
CFLAGS+= -I$(MINGWROOTDIR)/include -DWINVER=0x0501
CXXFLAGS+= -I$(MINGWROOTDIR)/include -DWINVER=0x0501
CXXFLAGS += -Wall -Wextra -Wno-deprecated
GCCFLAGS += -Wall -Wextra -Wno-deprecated
OPTS ?= -O3

LDFLAGS+= -lkernel32 -L $(MINGWROOTDIR)/lib
PARTLINKLD= $(LD)
BLOBGENLD= $(LD)
ifdef ONCYGWIN
C2GOTO_INCS=-I$(shell cygpath -w $(MINGWROOTDIR)/include | sed s/\\\\/\\\\\\\\/g) -D_MINGW
else
C2GOTO_INCS=-I$(MINGWROOTDIR)/include -D_MINGW
endif
WINDOWS= 1
MINGWRUNTIME= mingw_runtime_bodge
MINGWRUNOBJS=libstdc++-6.dll libgcc_s_sjlj-1.dll
PLATLIBS=-lws2_32 -lkernel32
FINLINKTAIL= $(LDFLAGS) $(LIBS) $(PLATLIBS)

# Work with latest version of gcc
GCCVERSIONGTEQ460 := $(shell expr 4.6.0 \<= `$(CC) -dumpversion`)
ifeq "$(GCCVERSIONGTEQ460)" "1"
  CXXFLAGS += -fpermissive
endif
