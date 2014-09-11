all: realall
ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc
MAINOBJNAME=langapi.o

SRCS = mode.cpp language_ui.cpp languages.cpp language_util.cpp

clean:
	-rm -f $(OBJS) $(OBJDIR)/.deps $(MAINOBJ)

include $(ESBMCDIR)/common

realall: $(OBJDIR)/.deps $(MAINOBJ)
