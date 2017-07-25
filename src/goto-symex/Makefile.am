noinst_LTLIBRARIES = libsymex.la
libsymex_la_SOURCES = symex_python.cpp \
      symex_target.cpp symex_target_equation.cpp symex_assign.cpp \
      symex_main.cpp goto_trace.cpp build_goto_trace.cpp \
      symex_function.cpp goto_symex_state.cpp symex_dereference.cpp \
      symex_goto.cpp builtin_functions.cpp slice.cpp symex_other.cpp \
      xml_goto_trace.cpp symex_valid_object.cpp \
      dynamic_allocation.cpp symex_catch.cpp renaming.cpp \
      execution_state.cpp reachability_tree.cpp witnesses.cpp \
      printf_formatter.cpp
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir)

symexincludedir = $(includedir)/goto-symex
symexinclude_HEADERS = build_goto_trace.h dynamic_allocation.h \
      execution_state.h goto_symex.h goto_symex_state.h goto_trace.h \
      reachability_tree.h renaming.h slice.h symex_target.h \
      symex_target_equation.h witnesses.h xml_goto_trace.h \
      printf_formatter.h

