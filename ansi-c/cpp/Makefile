OBJS=cpp.o cpy.o scanner.o compat.o
HEADERS=cpp.h

all: cpp

cpp: $(OBJS) $(HEADERS)
	$(CC) $(CFLAGS) $(LDFLAGS) $(OBJS) -o $@

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

cpp.o: cpy.o
