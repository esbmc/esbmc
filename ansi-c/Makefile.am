SUBDIRS = cpp headers
noinst_LIBRARIES = libansic.a libclib.a
libansic_a_SOURCES = \
      y.tab.cpp lex.yy.cpp \
      c_typecast.cpp ansi_c_parser.cpp \
      expr2c.cpp ansi_c_language.cpp c_sizeof.cpp c_main.cpp \
      c_types.cpp c_final.cpp trans_unit.cpp ansi_c_typecheck.cpp \
      c_link.cpp c_preprocess.cpp \
      c_typecheck_base.cpp c_typecheck_initializer.cpp \
      c_typecheck_typecast.cpp c_typecheck_code.cpp \
      c_typecheck_expr.cpp c_typecheck_type.cpp ansi_c_expr.cpp \
      unescape_string.cpp parse_float.cpp convert_float_literal.cpp \
      convert_integer_literal.cpp c_qualifiers.cpp \
      c_typecheck_argc_argv.cpp ansi_c_parse_tree.cpp \
      preprocessor_line.cpp convert_character_literal.cpp \
      convert_string_literal.cpp ansi_c_convert.cpp ansi_c_convert_type.cpp \
      type2name.cpp fix_symbol.cpp printf_formatter.cpp

noinst_PROGRAMS = c2goto
c2goto_SOURCES = c2goto.cpp cprover_blank_library.cpp $(libansic_a_SOURCES)
c2goto_LDADD = ../big-int/libbigint.a ../goto-programs/libgotoprograms.a ../langapi/liblangapi.a ../util/libutil.a cpp/libcpp.a ./headers/.objs/headers.o

libclib_a_SOURCES = cprover_library.cpp
libclib_a_LIBADD = clib32.o clib64.o

clib32.o: c2goto
	./c2goto library/*.c --output clib32.goto --no-lock-check
	./headers/flail.sh clib32_buf clib32.goto | $(CC) $(CFLAGS) -x c - -c -o $@

clib64.o: c2goto
	./c2goto library/*.c --output clib64.goto --no-lock-check
	./headers/flail.sh clib64_buf clib64.goto | $(CC) $(CFLAGS) -x c - -c -o $@
