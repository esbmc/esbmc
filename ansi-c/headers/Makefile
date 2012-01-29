ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/config.inc
MAINOBJ=$(OBJDIR)/headers.o

PLAINOBJS= stddef.o stdarg.o stdbool.o pthread.o pthreadtypes.o
OBJS= $(PLAINOBJS:%.o=$(OBJDIR)/%.o)

all: $(OBJDIR)/headers.o

$(OBJDIR)/%.o : %.h
	$(BLOBGENLD) $(LDFLAGS) -b binary $< -o $@ -r

clean:
	-rm $(OBJS) $(MAINOBJ) .deps

.PHONY: clean

include $(ESBMCDIR)/common
