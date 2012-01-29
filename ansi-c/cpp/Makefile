ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc

SRCS=cpp.c compat.c hooks.c
OBJS=$(OBJDIR)/y.tab.o $(OBJDIR)/lex.cpp.o
HEADERS=cpp.h

all: $(OBJDIR)/preproc.o

lex.cpp.c: scanner.l
	$(LEX) -Pcpp $(LFLAGS) $<

# Warcrime: yacc/bison won't produce y.tab.h unless lex/flex was already run :|
y.tab.c: cpy.y lex.cpp.c
	$(YACC) -d $(YFLAGS) -pcpp $<

# Another dependancy: can only build lex.cpp.o after bison has run to generate
# y.tab.h
$(OBJDIR)/lex.cpp.o: y.tab.c

cpp.c: y.tab.c # Includes y.tab.h

$(OBJS): $(HEADERS)

cpp.o: cpy.o

clean:
	-rm $(OBJDIR)/*.o
	-rm lex.cpp.c
	-rm y.tab.c
	-rm y.tab.h
	-rm y.output
	-rm .deps

ifdef WIN_MSVC
# For no reason I can detect, if opts are enabled in this dir, some linking
# conflict occurs with msvc and i2string.o. Increadibly bizare.
OPTS= /Od
endif

include $(ESBMCDIR)/common

$(OBJDIR)/preproc.o: $(OBJS)
	$(BLOBGENLD) $(LDFLAGS) $^ -o $@ -r
