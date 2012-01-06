ESBMCDIR= $(shell pwd)/../..

OBJS= stddef.o stdarg.o stdbool.o pthread.o pthreadtypes.o

all: headers.o

headers.o: $(OBJS)
	$(VCBINS)\\lib.exe $(LDFLAGS) $^ /OUT:$@

.h.o:
	x86_64-w64-mingw32-ld -b binary $< -o $@ -r

clean:
	-rm *.o .deps

.PHONY: clean

include $(ESBMCDIR)/common
