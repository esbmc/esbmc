ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc

SRCS=cpp.c compat.c hooks.c token.c
OBJS=$(OBJDIR)/y.tab.o
HOSTOBJS=$(HOSTOBJDIR)/y.tab.o
HEADERS=cpp.h

all: $(OBJDIR)/preproc.o

y.tab.c: cpy.y
	$(YACC) -d $(YFLAGS) -pcpp $<

cpp.c: y.tab.c # Includes y.tab.h

$(OBJS): $(HEADERS)

cpp.o: cpy.o

clean:
	-rm $(OBJDIR)/*.o
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

$(HOSTOBJDIR)/preproc.o: $(HOSTOBJS)
	$(HOSTBLOBGENLD) $(HOSTLDFLAGS) $^ -o $@ -r

$(OBJDIR)/preproc.o: $(OBJS) | $(HOSTOBJDIR)/preproc.o
	$(BLOBGENLD) $(LDFLAGS) $^ -o $@ -r
