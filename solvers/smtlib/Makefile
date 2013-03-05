ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJ= $(OBJDIR)/smtlib.o
all: $(OBJDIR)/.deps $(MAINOBJ)

SRCS= smtlib_conv.cpp lex.yy.cpp y.tab.cpp

lex.yy.cpp: smtlib.l y.tab.cpp
	$(LEX) -Psmtlib -olex.yy.cpp smtlib.l

y.tab.cpp: smtlib.y
	$(YACC) $(YFLAGS) $$flags -psmtlib -d smtlib.y -o y.tab.cpp

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/.deps


STRICTCOMPILE=1

include $(ESBMCDIR)/common
