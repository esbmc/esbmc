ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc
MAINOBJ= $(OBJDIR)/goto-symex.o
all: $(OBJDIR)/.deps $(MAINOBJ)

SRCS= symex_target.cpp symex_target_equation.cpp basic_symex.cpp \
      symex_main.cpp goto_trace.cpp build_goto_trace.cpp \
      symex_function.cpp goto_symex_state.cpp symex_dereference.cpp \
      symex_goto.cpp builtin_functions.cpp slice.cpp symex_other.cpp \
      slice_by_trace.cpp xml_goto_trace.cpp symex_valid_object.cpp \
      dynamic_allocation.cpp precondition.cpp postcondition.cpp \
      execution_state.cpp reachability_tree.cpp read_write_set.cpp \
      crypto_hash.cpp

clean:
	-rm -f $(OBJS) $(OBJDIR)/.deps $(MAINOBJ)

include $(ESBMCDIR)/common
