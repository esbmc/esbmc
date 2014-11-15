noinst_LTLIBRARIES = libsmtlib.la
libsmtlib_la_SOURCES = smtlib_conv.cpp smtlib.l smtlib.y
AM_YFLAGS = -d -psmtlib
AM_LFLAGS = --header-file=y.tab.hpp -Psmtlib -olex.yy.cpp
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -I$(top_srcdir)/util
