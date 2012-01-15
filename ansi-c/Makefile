ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc

SRCS= c_typecast.cpp ansi_c_parser.cpp \
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

OBJS= $(OBJDIR)/y.tab.o $(OBJDIR)/lex.yy.o

INCLUDES= -I .. -I ../util

all: $(OBJDIR)/.deps cpp_dir headers_dir $(OBJDIR)/ansi-c.o

###############################################################################

y.tab.cpp: parser.y
	$(YACC) $(YFLAGS) $$flags -pyyansi_c -d parser.y -o y.tab.cpp
	if [ -e y.tab.hpp ] ; then mv y.tab.hpp y.tab.h ; else \
        mv y.tab.cpp.h y.tab.h ; fi

lex.yy.cpp: scanner.l
	$(LEX) -Pyyansi_c -olex.yy.cpp scanner.l

# extra dependencies
$(OBJDIR)/y.tab.o: y.tab.cpp
# Blocking on y.tab.o ensures y.tab.h exists and has been moved.
$(OBJDIR)/lex.yy.o: lex.yy.cpp  $(OBJDIR)/y.tab.o

###############################################################################

C2GOTO_DEPS= $(OBJS) ../util/$(OBJDIR)/util.o ../langapi/$(OBJDIR)/langapi.o \
	     ../big-int/$(OBJDIR)/bigint.o \
	     ../goto-programs/$(OBJDIR)/goto-programs.o \
	     ../pointer-analysis/$(OBJDIR)/pointer-analysis.o \
	     cpp/preproc.o headers/headers.o

c2goto: c2goto.cpp $(OBJDIR)/cprover_blank_library.o | $(C2GOTO_DEPS)
	$(FINLINKBIN) $(CXXFLAGS) $^ $(FINLINKOUT) $@ $(C2GOTO_DEPS) $(FINLINKTAIL)

$(OBJDIR)/clib32.o: c2goto library/*.c
	./c2goto library/*.c --32 --output clib32.goto --no-lock-check $(C2GOTO_INCS)
	$(BLOBGENLD) $(LDFLAGS) -b binary clib32.goto -o $(OBJDIR)/clib32.o -r

$(OBJDIR)/clib64.o: c2goto library/*.c
	./c2goto library/*.c --32 --output clib64.goto --no-lock-check $(C2GOTO_INCS)
	$(BLOBGENLD) $(LDFLAGS) -b binary clib64.goto -o $(OBJDIR)/clib64.o -r

$(OBJDIR)/clib.o: c2goto $(OBJDIR)/clib32.o $(OBJDIR)/clib64.o
	$(PARTLINKLD) $(OBJDIR)/clib32.o $(OBJDIR)/clib64.o $(PARTLINKOUT)$(OBJDIR)/clib.o $(PARTLINKOPTS)

###############################################################################

include $(ESBMCDIR)/common

cpp/preproc.o: cpp_dir

cpp_dir:
	$(MAKE) -C cpp

.PHONY: cpp_dir

headers/headers.o: headers_dir

headers_dir:
	$(MAKE) -C headers

.PHONY: headers_dir

###############################################################################

$(OBJDIR)/ansi-c.o: $(OBJS) $(OBJDIR)/cprover_library.o $(OBJDIR)/clib.o cpp/preproc.o headers/headers.o
	$(PARTLINKLD) $(PARTLINKOUT)$@ $^ $(PARTLINKOPTS)

clean:
	$(MAKE) -C cpp clean
	$(MAKE) -C headers clean
	-rm -f $(MAINOBJ)
	-rm -f $(OBJS) y.tab.h y.tab.cpp lex.yy.cpp y.tab.cpp.output y.output
	-rm -f c2goto
	-rm -f *.goto
	-rm -f $(OBJDIR)/clib32.o $(OBJDIR)/clib64.o $(OBJDIR)/clib.o
	-rm -f $(OBJDIR)/cprover_blank_library.o $(OBJDIR)/cprover_library.o
	-rm $(OBJDIR)/.deps
