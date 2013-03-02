ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJ=$(OBJDIR)/headers.o

PLAINOBJS= stddef.o stdarg.o stdbool.o pthread.o pthreadtypes.o
OBJS= $(PLAINOBJS:%.o=$(OBJDIR)/%.o)
HOSTOBJS= $(PLAINOBJS:%.o=$(HOSTOBJDIR)/%.o)

all: $(OBJDIR)/headers.o

$(OBJDIR)/%.o : %.h
	$(BLOBGENLD) $(LDFLAGS) -b binary $< -o $@ -r

ifdef CROSSCOMPILE
$(HOSTOBJDIR)/%.o : %.h
	$(HOSTBLOBGENLD) $(HOSTLDFLAGS) -b binary $< -o $@ -r
endif

clean:
	-rm $(OBJS) $(MAINOBJ) .deps

.PHONY: clean

include $(ESBMCDIR)/common
