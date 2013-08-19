all: realall
ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJNAME=headers.o

PLAINOBJS= stddef.o stdarg.o stdbool.o pthread.o pthreadtypes.o digitalfilter.o
OBJS= $(PLAINOBJS:%.o=$(OBJDIR)/%.o)
HOSTOBJS= $(PLAINOBJS:%.o=$(HOSTOBJDIR)/%.o)

$(OBJDIR)/%.o : %.h
	./flail.sh "$(<:.h=)_buf" $< | $(CC) $(CFLAGS) -x c - -c -o $@

ifdef CROSSCOMPILE
$(HOSTOBJDIR)/%.o : %.h
	$(HOSTBLOBGENLD) $(HOSTLDFLAGS) -b binary $< -o $@ -r
endif

clean:
	-rm $(OBJS) $(MAINOBJ) .deps

.PHONY: clean

include $(ESBMCDIR)/common

realall: $(OBJDIR)/headers.o

distclean: clean
