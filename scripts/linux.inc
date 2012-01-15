CXXFLAGS += -Wall -Wextra -Wno-deprecated
GCCFLAGS += -Wall -Wextra -Wno-deprecated
OPTS ?= -O3

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
