include ../../common
include ../../config.inc

OBJS= stddef.o

all: headers.o

headers.o: $(OBJS)
	$(LD) $^ -o $@ -r

.h.o:
	$(LD) -b binary $< -o $@ -r
