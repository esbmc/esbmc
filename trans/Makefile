SRC = bmc_map.cpp counterexample.cpp unwind.cpp property.cpp get_trans.cpp \
      instantiate.cpp word_level_trans.cpp netlist.cpp \
      var_map.cpp unwind_netlist.cpp ldg.cpp compute_ct.cpp \
      trans_trace.cpp show_modules.cpp netlist_trans.cpp \
      map_aigs.cpp

OBJ = $(SRC:.cpp=.o)

INCLUDES= -I .. -I ../util

all: trans.o

include ../config.inc
include ../common

###############################################################################

trans.o: $(OBJ)
	$(LD) -r -o $@ $(OBJ)

clean:
	rm -f *.o
	-rm *.d
