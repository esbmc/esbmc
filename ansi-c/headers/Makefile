ESBMCDIR= $(shell pwd)/../..
MAINOBJ=headers.o

OBJS= stddef.o stdarg.o stdbool.o pthread.o pthreadtypes.o

all: headers.o

.h.o:
	$(BLOBGENLD) $(LDFLAGS) -b binary $< -o $@ -r

clean:
	-rm *.o .deps

.PHONY: clean

include $(ESBMCDIR)/common
