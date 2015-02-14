SUBDIRS = smt prop @ESBMC_SOLVER_DIRS@
noinst_LTLIBRARIES = libsolvers.la
libsolvers_la_SOURCES = solve.cpp
libsolvers_la_LIBADD = ./smt/libsmt.la ./prop/libpropproplibprop.la @ESBMC_SOLVER_BACKENDS@
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -I$(top_srcdir)/util
