ifdef TARGET64
MINGWID=amd64-mingw32msvc
Z3LIBDIR=x64
else
MINGWID=i586-mingw32msvc
Z3LIBDIR=lib
endif

ifdef USE_SOLVER_Z3
Z3 = $(SATDIR)/z3_win
CXXFLAGS+= -DZ3
INCLUDES+= -I $(Z3)/include
# Vastly irritating, is that mingw linking against z3.lib causes silent linking
# errors at runtime. So, link directly against the dll.
LIBS+= $(Z3)/$(Z3LIBDIR)/z3.dll
endif

MINGWROOTDIR= /usr/$(MINGWID)
MINGWBINDIR=$(MINGWROOTDIR)/bin

CC     = $(MINGWID)-gcc
CXX    = $(MINGWID)-g++
LD     = $(MINGWID)-ld
AR     = $(MINGWID)-ar
CFLAGS+= -I$(MINGWROOTDIR)/include -DWINVER=0x0501
CXXFLAGS+= -I$(MINGWROOTDIR)/include -DWINVER=0x0501
CXXFLAGS += -Wall -Wextra -Wno-deprecated
GCCFLAGS += -Wall -Wextra -Wno-deprecated
OPTS ?= -O3
DEPCXX = $(CXX)

PARTLINKLD= $(LD)
BLOBGENLD= $(LD)

C2GOTO_INCS=-I$(MINGWROOTDIR)/include
WINDOWS= 1
PLATLIBS=-lws2_32 -lkernel32
FINLINKTAIL= $(LDFLAGS) $(LIBS) $(PLATLIBS) -Wl,--stack,8388608

# Always add -static to the command line; this causes things like libstdc++ and
# the GCC runtime library to be linked in statically rather than dynamically,
# which means we don't have to cart around the mingw runtime libraries. And,
# they're public domain, so no GPL worries.
# This doesn't result in an actual static binary, because there's no such thing
# on windows, just no additional dependancies
FINLINKTAIL+= -static

HOSTCC=     gcc
HOSTCXX=    g++
HOSTDEPCXX= g++
HOSTLD=     ld
HOSTAR=     ar
HOSTLDFLAGS=
HOSTCXXFLAGS=$(INCLUDES) -fpermissive
HOSTCFLAGS=$(INCLUDES)
HOSTBLOBGENLD= ld
HOSTPARTLINKLD= ld
HOSTPARTLINKOUT= -o
HOSTPARTLINKOPTS= -r $(HOSTLDFLAGS)
HOSTFINLINKBIN= $(HOSTCXX)
HOSTFINLINKOUT= -o
HOSTFINLINKTAIL= $(HOSTCXXFLAGS) $(LIBS) -lgmp -ldl -lgcc
CROSSCOMPILE=1
