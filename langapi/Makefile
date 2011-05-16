SRC = mode.cpp language_ui.cpp languages.cpp language_util.cpp

OBJ = $(SRC:.cpp=.o)

INCLUDES= -I .. -I ../util

all: langapi.o dummies

include ../config.inc
include ../common

###############################################################################

langapi.o: $(OBJ)
	$(LD) $(LDFLAGS) -r -o langapi.o $(OBJ)
	
OBJ_DUMMY = $(patsubst %, %_dummy.o, $(LANGUAGES))
	
dummies: $(OBJ_DUMMY)

clean:
	rm -f *.o

