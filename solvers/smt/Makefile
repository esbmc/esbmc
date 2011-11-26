include ../../config.inc
include ../../common
include ../../local.inc

CXXFLAGS+= -I../.. -I../../util

OBJS = smt_dec.o smt_prop.o smt_conv.o

all: smt.o

smt.o: $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS)

clean:
	-rm *.o

.PHONY: clean
