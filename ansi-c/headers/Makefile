ESBMCDIR= $(shell pwd)/../..

OBJS= stddef.o stdarg.o stdbool.o

all: headers.o

headers.o: $(OBJS)
	$(LD) $(LDFLAGS) $^ -o $@ -r

.h.o:
	$(LD) $(LDFLAGS) -b binary $< -o $@ -r

clean:
	-rm *.o .deps

.PHONY: clean

include $(ESBMCDIR)/common
