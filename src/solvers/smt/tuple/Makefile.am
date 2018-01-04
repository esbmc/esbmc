noinst_LTLIBRARIES = libsmttuple.la
libsmttuple_la_SOURCES = smt_tuple_array_ast.cpp smt_tuple_node.cpp \
      smt_tuple_sym.cpp smt_tuple_node_ast.cpp smt_tuple_sym_ast.cpp

AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir)

smttupleincludedir = $(includedir)/solvers/smt/tuple
smttupleinclude_HEADERS = smt_tuple_array_ast.h smt_tuple_node_ast.h \
      smt_tuple_sort.h smt_tuple_sym.h smt_tuple.h smt_tuple_node.h \
      smt_tuple_sym_ast.h

