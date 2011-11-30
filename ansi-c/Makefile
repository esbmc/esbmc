ESBMCDIR= $(shell pwd)/..

SRCS= c_typecast.cpp y.tab.cpp lex.yy.cpp ansi_c_parser.cpp \
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

INCLUDES= -I .. -I ../util

include $(ESBMCDIR)/common

all: cpp_dir headers_dir ansi-c.o

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

C2GOTO_DEPS= $(OBJS) ../util/util.o ../langapi/langapi.o ../big-int/bigint.o \
	../goto-programs/goto-programs.o ../pointer-analysis/pointer-analysis.o\
	cpp/preproc.o headers/headers.o

c2goto: c2goto.cpp cprover_blank_library.o | $(C2GOTO_DEPS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(C2GOTO_DEPS)

clib32.o: c2goto library/*.c
	./c2goto library/*.c --32 --output clib32.goto --no-lock-check
	$(LD) $(LDFLAGS) -b binary clib32.goto -o clib32.o -r

clib64.o: c2goto library/*.c
	./c2goto library/*.c --32 --output clib64.goto --no-lock-check
	$(LD) $(LDFLAGS) -b binary clib64.goto -o clib64.o -r

clib.o: c2goto clib32.o clib64.o
	$(LD) $(LDFLAGS) clib32.o clib64.o -o clib.o -r

###############################################################################

cpp/preproc.o: cpp_dir

cpp_dir:
	$(MAKE) -C cpp

.PHONY: cpp_dir

headers/headers.o: headers_dir

headers_dir:
	$(MAKE) -C headers

.PHONY: headers_dir

###############################################################################

ansi-c.o: $(OBJS) cprover_library.o clib.o cpp/preproc.o headers/headers.o
	$(LD) $(LDFLAGS) -r -o $@ $^

clean:
	$(MAKE) -C cpp clean
	$(MAKE) -C headers clean
	-rm -f $(OBJS) *.o y.tab.h y.tab.cpp lex.yy.cpp y.tab.cpp.output y.output
	-rm -f c2goto
	-rm -f *.goto
	-rm .deps
