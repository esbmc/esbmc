SRC = mode.cpp language_ui.cpp languages.cpp language_util.cpp

OBJ = $(SRC:.cpp=.o)

INCLUDES= -I .. -I ../util

all: langapi.o

include ../config.inc
include ../common

###############################################################################

langapi.o: $(OBJ)
	$(LD) $(LDFLAGS) -r -o langapi.o $(OBJ)
	
clean:
	rm -f *.o
	rm -f *.d

