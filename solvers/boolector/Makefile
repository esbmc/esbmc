include ../../config.inc
include ../../common
include ../../local.inc

CXXFLAGS+= -I../.. -I../../util -I$(BOOLECTOR)/include

OBJS= boolector_dec.o boolector_prop.o boolector_conv.o boolector_get.o

all: boolector.o

boolector.o: $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS)

clean:
	-rm *.o
	-rm *.d

.PHONY: clean
