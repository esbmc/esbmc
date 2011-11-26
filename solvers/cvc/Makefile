include ../../config.inc
include ../../common
include ../../local.inc

CXXFLAGS+= -I../.. -I../../util

OBJS = cvc_prop.o cvc_conv.o cvc_dec.o

all: cvc.o

cvc.o: $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS)

clean:
	-rm *.o

.PHONY: clean
