ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/common

SRCS = mode.cpp language_ui.cpp languages.cpp language_util.cpp

INCLUDES= -I .. -I ../util

all: langapi.o

###############################################################################

langapi.o: $(OBJS)
	$(LD) $(LDFLAGS) -r -o langapi.o $(OBJS)
	
clean:
	rm -f *.o
	rm -f *.d

