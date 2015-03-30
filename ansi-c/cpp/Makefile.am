noinst_LIBRARIES = libcpp.a
libcpp_a_SOURCES = cpy.y cpp.c compat.c hooks.c token.c
AM_YFLAGS = -pcpp -d
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -I$(top_srcdir)/utils
BUILT_SOURCES = cpy.h
