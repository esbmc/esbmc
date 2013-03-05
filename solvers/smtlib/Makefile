ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJ= $(OBJDIR)/smtlib.o
all: $(OBJDIR)/.deps $(MAINOBJ)

SRCS= smtlib_conv.cpp lex.yy.cpp y.tab.cpp

lex.yy.cpp: scanner.l
	$(LEX) -Psmtlib -olex.yy.c scanner.l

y.tab.cpp: parser.y
	$(YACC) $(YFLAGS) $$flags -psmtlib -d parser.y -o y.tab.cpp

clean:
	-rm -f $(MAINOBJ) $(OBJS) $(OBJDIR)/.deps


STRICTCOMPILE=1

include $(ESBMCDIR)/common
