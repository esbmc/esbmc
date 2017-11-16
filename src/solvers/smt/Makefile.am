noinst_LTLIBRARIES = libsmt.la
libsmt_la_SOURCES = fp_conv.cpp array_conv.cpp smt_byteops.cpp smt_casts.cpp \
      smt_conv.cpp smt_memspace.cpp smt_overflow.cpp smt_tuple_node.cpp \
      smt_tuple_sym.cpp
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir)

smtincludedir = $(includedir)/solvers/smt
smtinclude_HEADERS = array_conv.h smt_array.h smt_conv.h smt_tuple.h \
      smt_tuple_flat.h

