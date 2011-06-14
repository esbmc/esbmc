include ../../common
include ../../config.inc

OBJS= stddef.o

all: $(OBJS)

.h.o:
	$(LD) -b binary $< -o $@ -r
