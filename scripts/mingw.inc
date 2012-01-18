ifdef TARGET64
MINGWID=x86_64-w64-mingw32
Z3LIBDIR=x64
else
MINGWID=i686-pc-mingw32
Z3LIBDIR=lib
endif

MINGWROOTDIR= /usr/$(MINGWID)/sys-root/mingw
MINGWBINDIR=$(MINGWROOTDIR)/bin

ifndef ONCYGWIN
ifdef TARGET64
# There are no fedora packages for 64 bit mingw, therefore there's no standard
# location (unless you can show me a distribution that does have such packages).
# So, if we have this target (linux, 64 bit, mingw) then assume that what we
# have is a minw32-w64 tarball extracted to /usr/$(MINGWID).
MINGWROOTDIR=/usr/x86_64-w64-mingw32/x86_64-w64-mingw32
# Also, by default libstdc++ is in lib. Which is, after all, correct.
MINGWBINDIR=$(MINGWROOTDIR)/lib
endif
endif

CC     = $(MINGWID)-gcc
CXX    = $(MINGWID)-g++
LD     = $(MINGWID)-ld
AR     = $(MINGWID)-ar
CFLAGS+= -I$(MINGWROOTDIR)/include -DWINVER=0x0501
CXXFLAGS+= -I$(MINGWROOTDIR)/include -DWINVER=0x0501
CXXFLAGS += -Wall -Wextra -Wno-deprecated
GCCFLAGS += -Wall -Wextra -Wno-deprecated
OPTS ?= -O3

PARTLINKLD= $(LD)
BLOBGENLD= $(LD)

ifndef MSVC_10_HEADERS
include define_MSVC_10_HEADERS_please
# It should locate where the C library headers from Visual Studio 10 are.
endif

ifdef ONCYGWIN
C2GOTO_INCS=-I$(shell cygpath -w $(MSVC_10_HEADERS)/include | sed s/\\\\/\\\\\\\\/g) -D_MINGW
else
C2GOTO_INCS=-I$(MSVC_10_HEADERS)/include -D_MINGW
endif
WINDOWS= 1
PLATLIBS=-lws2_32 -lkernel32
FINLINKTAIL= $(LDFLAGS) $(LIBS) $(PLATLIBS)

# Always add -static to the command line; this causes things like libstdc++ and
# the GCC runtime library to be linked in statically rather than dynamically,
# which means we don't have to cart around the mingw runtime libraries. And,
# they're public domain, so no GPL worries.
# This doesn't result in an actual static binary, because there's no such thing
# on windows, just no additional dependancies
FINLINKTAIL+= -static

# Work with latest version of gcc
GCCVERSIONGTEQ460 := $(shell expr 4.6.0 \<= `$(CC) -dumpversion`)
ifeq "$(GCCVERSIONGTEQ460)" "1"
  CXXFLAGS += -fpermissive
endif
