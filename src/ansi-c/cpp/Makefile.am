noinst_LTLIBRARIES = libcpp.la
libcpp_la_SOURCES = cpy.y cpp.c compat.c hooks.c token.c
AM_YFLAGS = -pcpp -d
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -I$(top_srcdir)/utils
BUILT_SOURCES = cpy.h
