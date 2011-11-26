SRC = value_set.cpp goto_program_dereference.cpp value_set_analysis.cpp \
      dereference.cpp pointer_offset_sum.cpp add_failed_symbols.cpp \
      show_value_sets.cpp value_set_domain.cpp \
      value_set_analysis_fi.cpp value_set_fi.cpp value_set_domain_fi.cpp \
      value_set_analysis_fivr.cpp value_set_fivr.cpp value_set_domain_fivr.cpp \
      value_set_analysis_fivrns.cpp value_set_fivrns.cpp value_set_domain_fivrns.cpp

OBJ = $(SRC:.cpp=.o)

INCLUDES= -I .. -I ../util

all: pointer-analysis.o

include ../config.inc
include ../common

###############################################################################

pointer-analysis.o: $(OBJ)
	$(LD) $(LDFLAGS) -r -o pointer-analysis.o $(OBJ)

clean:
	rm -f *.o

