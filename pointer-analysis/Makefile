ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc
MAINOBJ= $(OBJDIR)/pointer-analysis.o
all: $(OBJDIR)/.deps $(MAINOBJ)

SRCS= value_set.cpp goto_program_dereference.cpp value_set_analysis.cpp \
      dereference.cpp pointer_offset_sum.cpp add_failed_symbols.cpp \
      show_value_sets.cpp value_set_domain.cpp

clean:
	-rm -f $(OBJS) $(OBJDIR)/.deps $(MAINOBJ)

include $(ESBMCDIR)/common
