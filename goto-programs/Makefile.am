noinst_LIBRARIES = libgotoprograms.a
libgotoprograms_a_SOURCES = \
      goto_convert.cpp goto_function.cpp goto_main.cpp goto_sideeffects.cpp \
      goto_program.cpp goto_check.cpp goto_inline.cpp \
      remove_skip.cpp goto_convert_functions.cpp \
      builtin_functions.cpp show_claims.cpp destructor.cpp set_claims.cpp \
      add_race_assertions.cpp rw_set.cpp read_goto_binary.cpp \
      static_analysis.cpp string_abstraction.cpp \
      goto_program_serialization.cpp goto_function_serialization.cpp \
      read_bin_goto_object.cpp goto_program_irep.cpp \
      format_strings.cpp loop_numbers.cpp \
      write_goto_binary.cpp
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -I$(top_srcdir)/util -Wno-unused-private-field
