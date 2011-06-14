include ../../common
include ../../config.inc

OBJS=cpy.o cpp.o scanner.o compat.o
HEADERS=cpp.h

all: preproc.o

preproc.o: $(OBJS) $(HEADERS)
	$(LD) $(LDFLAGS) $(OBJS) -r -o $@

.c.o:
	$(CC) $(DEFS) $(CPPFLAGS) $(CFLAGS) -c $<

.l.o:
	$(LEX) $(LFLAGS) $<
	$(CC) $(DEFS) $(CPPFLAGS) $(CFLAGS) -c -o $@ lex.yy.c

.y.o:
	$(YACC) -d $(YFLAGS) $<
	$(CC) $(DEFS) $(CPPFLAGS) $(CFLAGS) -c -o $@ y.tab.c

compat.o: compat.c
	$(CC) $(DEFS) $(CPPFLAGS) $(CFLAGS) -c -o $@ compat.c

$(OBJS): $(HEADERS)

clean:
	-rm *.o
	-rm lex.yy.c
	-rm y.tab.c
	-rm y.tab.h
