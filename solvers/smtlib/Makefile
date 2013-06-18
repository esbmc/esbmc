all: realall
ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJNAME=smtlib.o

SRCS= smtlib_conv.cpp lex.yy.cpp y.tab.cpp

lex.yy.cpp: smtlib.l y.tab.cpp
	$(LEX) -Psmtlib -olex.yy.cpp smtlib.l

y.tab.cpp: y.tab.hpp

y.tab.hpp: smtlib.y
	$(YACC) $(YFLAGS) $$flags -psmtlib -d smtlib.y -o y.tab.cpp

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/.deps lex.yy.cpp y.tab.cpp y.tab.hpp y.output


STRICTCOMPILE=1

include $(ESBMCDIR)/common

realall: $(OBJDIR)/.deps $(MAINOBJ)
