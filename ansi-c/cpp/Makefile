ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc

SRCS=cpp.c compat.c hooks.c
OBJS=$(OBJDIR)/y.tab.o $(OBJDIR)/lex.cpp.o
HEADERS=cpp.h

all: preproc.o

lex.cpp.c: scanner.l
	$(LEX) -Pcpp $(LFLAGS) $<

# Warcrime: yacc/bison won't produce y.tab.h unless lex/flex was already run :|
y.tab.c: cpy.y lex.cpp.c
	$(YACC) -d $(YFLAGS) -pcpp $<

cpp.c: y.tab.c # Includes y.tab.h

compat.o: compat.c
	$(CC) $(DEFS) $(CPPFLAGS) $(CFLAGS) -c -o $@ compat.c

$(OBJS): $(HEADERS)

cpp.o: cpy.o

clean:
	-rm $(OBJDIR)/*.o
	-rm lex.cpp.c
	-rm y.tab.c
	-rm y.tab.h
	-rm y.output
	-rm .deps

include $(ESBMCDIR)/common

preproc.o: $(OBJS)
	$(BLOBGENLD) $^ -o $@ -r
