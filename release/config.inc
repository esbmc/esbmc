CC     = gcc
CXX    = g++
LD     = ld
AR     = ar
YACC   = bison -y
YFLAGS = -v
LEX    = flex

OPTS= -O3
GCCFLAGS= -Wall -Wextra -Wno-deprecated

# Work with latest version of gcc
GCCVERSIONGTEQ460 := $(shell expr 4.6.0 \<= `gcc -dumpversion`)
ifeq "$(GCCVERSIONGTEQ460)" "1"
  CXXFLAGS += -fpermissive
endif

OBJDIR= .objs
# Danger: evaluating expression creates directory :o
CREATEOBJDIR := $(shell mkdir $(OBJDIR) 2>/dev/null)

# Build with C++.
MODULE_CPP = 1
