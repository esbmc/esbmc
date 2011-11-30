ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc
MAINOBJ= $(OBJDIR)/langapi.o
all: $(MAINOBJ)

SRCS = mode.cpp language_ui.cpp languages.cpp language_util.cpp

clean:
	-rm -f $(OBJS) .deps

include $(ESBMCDIR)/common
