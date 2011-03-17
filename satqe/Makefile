SRC = cube_set.cpp cubes.cpp satqe_satcheck.cpp
OBJ = $(SRC:.cpp=.o)

LIBS =

all: satqe.a

include ../config.inc
include ../common

ifneq ($(MINISAT),)
MINISAT_INCLUDE=-I ../$(MINISAT)
endif

INCLUDES= $(MINISAT_INCLUDE) -I .. -I ../util

###############################################################################

satqe.a: $(OBJ)
	$(LD) -r -o satqe.a $(OBJ)

clean:
	rm -f $(OBJ) *.o satqe.a
