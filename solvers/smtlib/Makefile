ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJ= $(OBJDIR)/smtlib.o
all: $(OBJDIR)/.deps $(MAINOBJ)

SRCS= smtlib_conv.cpp lex.yy.cpp

lex.yy.cpp: scanner.l
	$(LEX) -Psmtlib -olex.yy.cpp scanner.l

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/.deps


STRICTCOMPILE=1

include $(ESBMCDIR)/common
