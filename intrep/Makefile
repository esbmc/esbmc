SRC = irep_language.cpp
OBJ = irep_language.o

INCLUDES= -I .. -I ../util

all: intrep.o

include ../config.inc
include ../common

###############################################################################

intrep.o: $(OBJ)
	$(LD) $(LDFLAGS) -r -o intrep.o $(OBJ)

clean:
	rm -f *.o *.d
