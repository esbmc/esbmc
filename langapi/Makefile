ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc
MAINOBJ= $(OBJDIR)/langapi.o
all: $(OBJDIR)/.deps $(MAINOBJ)

SRCS = mode.cpp language_ui.cpp languages.cpp language_util.cpp

clean:
	-rm -f $(OBJS) $(OBJDIR)/.deps $(MAINOBJ)

include $(ESBMCDIR)/common
