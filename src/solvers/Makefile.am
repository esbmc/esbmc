SUBDIRS = smt prop @ESBMC_SOLVER_DIRS@
noinst_LTLIBRARIES = libsolvers.la
libsolvers_la_SOURCES = solve.cpp smt_python.cpp
libsolvers_la_DEPENDENCIES = ./smt/libsmt.la ./prop/libpropproplibprop.la \
      @ESBMC_SOLVER_BACKENDS@
libsolvers_la_LIBADD = ./smt/libsmt.la ./prop/libpropproplibprop.la \
      @ESBMC_SOLVER_BACKENDS@
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir)

solversincludedir = $(includedir)/solvers
solversinclude_HEADERS = smt_python.h solve.h

