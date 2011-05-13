include ../../config.inc
include ../../common
include ../../local.inc

CPLUSFLAGS+= -I../.. -I../../util

OBJS = dplib_dec.o dplib_prop.o dplib_conv.o

all: dplib.o

dplib.o: $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS)

clean:
	-rm *.o

.PHONY: clean

