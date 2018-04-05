BUILT_SOURCES = smtlib.hpp smtlib_tok.cpp
noinst_LTLIBRARIES = libsmtlib.la
libsmtlib_la_SOURCES = smtlib.ypp smtlib_tok.lpp smtlib_conv.cpp
AM_YFLAGS = -d -psmtlib
AM_LFLAGS = --header-file=smtlib_tok.hpp -Psmtlib_tok -osmtlib_tok.cpp
# Use warnings because:
# smtlib_tok.cpp:728:18: error: comparison between signed and unsigned integer expressions [-Werror=sign-compare]
AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -Wno-sign-compare

smtlibincludedir = $(includedir)/solvers/smtlib
smtlibinclude_HEADERS = smtlib_conv.h

clean-local:
	-rm smtlib.cpp smtlib.hpp smtlib_tok.cpp smtlib_tok.hpp
