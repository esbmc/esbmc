noinst_LTLIBRARIES = libpointeranalysis.la
libpointeranalysis_la_SOURCES = value_set.cpp goto_program_dereference.cpp \
      value_set_analysis.cpp dereference.cpp show_value_sets.cpp \
      value_set_domain.cpp
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -I$(top_srcdir)/util

pointerincludedir = $(includedir)/pointer-analysis
pointerinclude_HEADERS = dereference.h goto_program_dereference.h \
      object_numbering.h show_value_sets.h value_set.h value_set_analysis.h \
      value_set_domain.h value_sets.h

