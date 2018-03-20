SUBDIRS = tuple fp

noinst_LTLIBRARIES = libsmt.la
libsmt_la_SOURCES = array_conv.cpp smt_byteops.cpp smt_casts.cpp \
      smt_conv.cpp smt_memspace.cpp smt_overflow.cpp
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir)

smtincludedir = $(includedir)/solvers/smt
smtinclude_HEADERS = array_conv.h smt_ast.h smt_conv.h smt_sort.h \
      smt_array.h

