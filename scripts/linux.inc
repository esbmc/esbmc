CXXFLAGS += -Wall -Wextra -Wno-deprecated
GCCFLAGS += -Wall -Wextra -Wno-deprecated
OPTS ?= -O3
PLATLIBS= -lgmp -ldl

ifdef TARGET32
CXXFLAGS += -m32
GCCFLAGS += -m32
LDFLAGS += -m elf_i386
endif

# Work with latest version of gcc
GCCVERSIONGTEQ460 := $(shell expr 4.6.0 \<= `$(CC) -dumpversion`)
ifeq "$(GCCVERSIONGTEQ460)" "1"
  CXXFLAGS += -fpermissive
endif

ifdef STATICLINK
FINLINKTAIL+= -static
endif

ifdef LINUXCOMPAT
CC=gcc34
CXX=g++34
else
# If not in compat mode, link against libgomp. Can't do it with g++34 for
# an unknown reason.
PLATLIBS+= -lgomp
endif
