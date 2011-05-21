SRC = c_typecast.cpp y.tab.cpp lex.yy.cpp ansi_c_parser.cpp \
      expr2c.cpp ansi_c_language.cpp c_sizeof.cpp c_main.cpp \
      c_types.cpp c_final.cpp trans_unit.cpp ansi_c_typecheck.cpp \
      c_link.cpp c_preprocess.cpp c_link_type_eq.cpp \
      c_typecheck_base.cpp c_typecheck_initializer.cpp \
      c_typecheck_typecast.cpp c_typecheck_code.cpp \
      c_typecheck_expr.cpp c_typecheck_type.cpp ansi_c_expr.cpp \
      unescape_string.cpp parse_float.cpp convert_float_literal.cpp \
      convert_integer_literal.cpp c_qualifiers.cpp \
      c_typecheck_argc_argv.cpp ansi_c_parse_tree.cpp \
      preprocessor_line.cpp convert_character_literal.cpp \
      convert_string_literal.cpp ansi_c_convert.cpp ansi_c_convert_type.cpp \
      type2name.cpp fix_symbol.cpp printf_formatter.cpp \

OBJ = $(SRC:.cpp=.o)

INCLUDES= -I .. -I ../util

all: ansi-c.o

include ../config.inc
include ../common

###############################################################################

y.tab.cpp: parser.y
	$(YACC) $(YFLAGS) $$flags -pyyansi_c -d parser.y -o y.tab.cpp

y.tab.h: y.tab.cpp
	if [ -e y.tab.hpp ] ; then mv y.tab.hpp y.tab.h ; else \
        mv y.tab.cpp.h y.tab.h ; fi

lex.yy.cpp: scanner.l
	$(LEX) -Pyyansi_c -olex.yy.cpp scanner.l

# extra dependencies
y.tab.o: y.tab.cpp y.tab.h
lex.yy.o: y.tab.cpp lex.yy.cpp y.tab.h

###############################################################################

cprover_library.o: cprover_library.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@ -c

cprover_blank_library.o: cprover_library.cpp
	$(CXX) $(CXXFLAGS) -DNO_CPROVER_LIBRARY $^ -o $@ -c

c2goto: $(OBJ) c2goto.cpp cprover_blank_library.o ../util/util.o \
		../langapi/langapi.o \
		../big-int/bigint.o \
		../langapi/csp_dummy.o \
		../langapi/intrep_dummy.o \
		../langapi/pvs_dummy.o \
		../langapi/smvlang_dummy.o \
		../langapi/promela_dummy.o \
		../langapi/verilog_dummy.o \
		../langapi/vhdl_dummy.o \
		../langapi/smtlang_dummy.o \
		../langapi/netlist_dummy.o \
		../langapi/pascal_dummy.o \
		../langapi/bplang_dummy.o \
		../langapi/simplifylang_dummy.o \
		../langapi/csharp_dummy.o \
		../langapi/nsf_dummy.o \
		../langapi/mdllang_dummy.o \
		../langapi/specc_dummy.o \
		../langapi/cpp_dummy.o \
		../langapi/php_dummy.o \
		../cvclang/cvclang.o \
		../goto-programs/goto-programs.o \
		../pointer-analysis/pointer-analysis.o
	$(CXX) $(CXXFLAGS) $^ -o $@

clib16.o: c2goto library
	./c2goto library/*.i --16 --output clib16.goto
	$(LD) -b binary clib16.goto -o clib16.o -r

clib32.o: c2goto library
	./c2goto library/*.i --32 --output clib32.goto
	$(LD) -b binary clib32.goto -o clib32.o -r

clib64.o: c2goto library
	./c2goto library/*.i --64 --output clib64.goto
	$(LD) -b binary clib64.goto -o clib64.o -r

library: library/*.c
	$(MAKE) -C library

clib.o: c2goto clib16.o clib32.o clib64.o
	$(LD) clib16.o clib32.o clib64.o -o clib.o -r

###############################################################################

ansi-c.o: $(OBJ) cprover_library.o clib.o
	$(LD) $(LDFLAGS) -r -o $@ $^

clean:
	$(MAKE) -C library clean
	-rm -f $(OBJ) *.o y.tab.h y.tab.cpp lex.yy.cpp y.tab.cpp.output y.output
	-rm -f c2goto
	-rm -f *.goto
