noinst_LIBRARIES = libsymex.a
libsymex_a_SOURCES = \
      symex_target.cpp symex_target_equation.cpp symex_assign.cpp \
      symex_main.cpp goto_trace.cpp build_goto_trace.cpp \
      symex_function.cpp goto_symex_state.cpp symex_dereference.cpp \
      symex_goto.cpp builtin_functions.cpp slice.cpp symex_other.cpp \
      xml_goto_trace.cpp symex_valid_object.cpp \
      dynamic_allocation.cpp symex_catch.cpp renaming.cpp \
      execution_state.cpp reachability_tree.cpp VarMap.cpp
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -I$(top_srcdir)/util
