noinst_LTLIBRARIES = libgotoprograms.la
libgotoprograms_la_SOURCES = \
      goto_convert.cpp goto_function.cpp goto_main.cpp goto_sideeffects.cpp \
      goto_program.cpp goto_check.cpp goto_inline.cpp \
      remove_skip.cpp goto_convert_functions.cpp remove_unreachable.cpp \
      builtin_functions.cpp show_claims.cpp destructor.cpp set_claims.cpp \
      add_race_assertions.cpp rw_set.cpp read_goto_binary.cpp \
      static_analysis.cpp \
      goto_program_serialization.cpp goto_function_serialization.cpp \
      read_bin_goto_object.cpp goto_program_irep.cpp \
      format_strings.cpp loop_numbers.cpp goto_loops.cpp \
      write_goto_binary.cpp goto_unwind.cpp goto_k_induction.cpp \
      loopst.cpp goto_python.cpp
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir)

gotoincludedir = $(includedir)/goto-programs
gotoinclude_HEADERS = add_race_assertions.h destructor.h format_strings.h \
      goto_check.h goto_convert_class.h goto_convert_functions.h \
      goto_function_serialization.h goto_functions.h goto_inline.h \
      goto_k_induction.h goto_loops.h goto_program.h goto_program_irep.h \
      goto_program_serialization.h goto_unwind.h loop_numbers.h loopst.h \
      read_bin_goto_object.h read_goto_binary.h \
      remove_skip.h remove_unreachable.h rw_set.h set_claims.h \
      show_claims.h static_analysis.h write_goto_binary.h

