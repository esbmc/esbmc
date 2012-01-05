ESBMCDIR= $(shell pwd)/../..

OBJS=cpy.o cpp.o scanner.o compat.o hooks.o
HEADERS=cpp.h

all: preproc.o

preproc.o: $(OBJS) $(HEADERS)
	$(LD) $(LDFLAGS) $(OBJS) -r -o $@

lex.cpp.c: scanner.l
	$(LEX) -Pcpp $(LFLAGS) $<

y.tab.c: cpy.y
	$(YACC) -d $(YFLAGS) -pcpp $<

y.tab.h: cpy.o

scanner.l: y.tab.h

compat.o: compat.c
	$(CC) $(DEFS) $(CPPFLAGS) $(CFLAGS) -c -o $@ compat.c

$(OBJS): $(HEADERS)

cpp.o: cpy.o

clean:
	-rm *.o
	-rm lex.cpp.c
	-rm y.tab.c
	-rm y.tab.h
	-rm y.output
	-rm .deps

include $(ESBMCDIR)/common
