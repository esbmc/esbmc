ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc
MAINOBJ= $(OBJDIR)/goto-programs.o
all: $(OBJDIR)/.deps $(MAINOBJ)

SRCS= goto_convert.cpp goto_function.cpp goto_main.cpp goto_sideeffects.cpp \
      goto_program.cpp goto_check.cpp \
      goto_function_pointers.cpp goto_functions.cpp goto_inline.cpp \
      remove_skip.cpp goto_convert_functions.cpp string_instrumentation.cpp \
      builtin_functions.cpp show_claims.cpp destructor.cpp set_claims.cpp \
      invariant_set.cpp \
      add_race_assertions.cpp rw_set.cpp read_goto_binary.cpp \
      static_analysis.cpp string_abstraction.cpp \
      goto_program_serialization.cpp goto_function_serialization.cpp \
      read_bin_goto_object.cpp goto_program_irep.cpp \
      flow_insensitive_analysis.cpp \
      format_strings.cpp loop_numbers.cpp \
      goto_program_template.cpp write_goto_binary.cpp

clean:
	-rm -f $(OBJS) $(OBJDIR)/.deps $(MAINOBJ)

include $(ESBMCDIR)/common
