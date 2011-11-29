ESBMCDIR= $(shell pwd)/../..
include $(ESBMCDIR)/common

OBJS= stddef.o stdarg.o stdbool.o

all: headers.o

headers.o: $(OBJS)
	$(LD) $(LDFLAGS) $^ -o $@ -r

.h.o:
	$(LD) $(LDFLAGS) -b binary $< -o $@ -r

clean:
	-rm *.o
	-rm *.d

.PHONY: clean
