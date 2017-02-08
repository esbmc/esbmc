SUBDIRS = cpp

noinst_LTLIBRARIES = libansicscanner.la libansic.la

AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir) -I$(top_srcdir)/util
AM_CFLAGS = $(ESBMC_CFLAGS) -I$(top_srcdir) -I$(top_srcdir)/util
AM_YFLAGS = -d -pyyansi_c
AM_LFLAGS = --header-file=scanner.hpp -Pyyansi_c -o scanner.cpp

libansicscanner_la_SOURCES = scanner.lpp
libansicscanner_la_CXXFLAGS = $(ESBMC_CXXFLAGS_WARNINGS) -I$(top_srcdir) -I$(top_srcdir)/util

libansic_la_SOURCES = parser.ypp \
      c_typecast.cpp ansi_c_parser.cpp \
      expr2c.cpp ansi_c_language.cpp c_sizeof.cpp c_main.cpp \
      c_final.cpp trans_unit.cpp ansi_c_typecheck.cpp \
      c_link.cpp c_preprocess.cpp \
      c_typecheck_base.cpp c_typecheck_initializer.cpp \
      c_typecheck_typecast.cpp c_typecheck_code.cpp \
      c_typecheck_expr.cpp c_typecheck_type.cpp \
      unescape_string.cpp parse_float.cpp convert_float_literal.cpp \
      convert_integer_literal.cpp c_qualifiers.cpp \
      c_typecheck_argc_argv.cpp ansi_c_parse_tree.cpp \
      preprocessor_line.cpp convert_character_literal.cpp \
      convert_string_literal.cpp ansi_c_convert.cpp ansi_c_convert_type.cpp \
      type2name.cpp fix_symbol.cpp printf_formatter.cpp
libansic_la_LIBADD = libansicscanner.la
BUILT_SOURCES = parser.hpp

clean-local:
	-rm scanner.cpp scanner.hpp parser.hpp parser.cpp
.PHONY: clean-local
