SUBDIRS = cpp

noinst_LTLIBRARIES = libansicscanner.la libansic.la

AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir)
AM_CFLAGS = $(ESBMC_CFLAGS) -I$(top_srcdir)
AM_YFLAGS = -d -pyyansi_c
AM_LFLAGS = --header-file=scanner.hpp -Pyyansi_c -o scanner.cpp

libansicscanner_la_SOURCES = scanner.lpp
libansicscanner_la_CXXFLAGS = $(ESBMC_CXXFLAGS_WARNINGS) -I$(top_srcdir)

libansic_la_SOURCES = parser.ypp \
      ansi_c_parser.cpp \
      ansi_c_language.cpp c_main.cpp \
      c_final.cpp trans_unit.cpp ansi_c_typecheck.cpp \
      c_preprocess.cpp type2name.cpp \
      c_typecheck_base.cpp c_typecheck_initializer.cpp \
      c_typecheck_typecast.cpp c_typecheck_code.cpp \
      c_typecheck_expr.cpp c_typecheck_type.cpp \
      unescape_string.cpp parse_float.cpp convert_float_literal.cpp \
      convert_integer_literal.cpp \
      c_typecheck_argc_argv.cpp ansi_c_parse_tree.cpp \
      preprocessor_line.cpp convert_character_literal.cpp \
      convert_string_literal.cpp ansi_c_convert.cpp ansi_c_convert_type.cpp 
libansic_la_LIBADD = libansicscanner.la
BUILT_SOURCES = parser.hpp

ansicincludedir = $(includedir)/ansi-c
ansicinclude_HEADERS = ansi_c_convert.h ansi_c_convert_type.h \
      ansi_c_declaration.h ansi_c_language.h ansi_c_parse_tree.h \
      ansi_c_parser.h ansi_c_typecheck.h c_final.h c_main.h \
      c_preprocess.h c_storage_spec.h \
      c_typecheck_base.h convert_character_literal.h convert_float_literal.h \
      convert_integer_literal.h convert_string_literal.h \
      gcc_builtin_headers.h headers.h parse_float.h \
      preprocessor_line.h trans_unit.h \
      unescape_string.h type2name.h

clean-local:
	-rm scanner.cpp scanner.hpp parser.hpp parser.cpp
.PHONY: clean-local
