include ../../config.inc
include ../../common

CPLUSFLAGS+= -I../.. -I ../../util

OBJS= prop.o prop_conv.o prop_conv_store.o aig_formula.o aig.o aig_prop.o

all: propobj.o

propobj.o: $(OBJS)
	$(LD) -r -o $@ $^ $(LDFLAGS)

clean:
	-rm *.o

.PHONY: clean
