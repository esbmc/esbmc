ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc

SRCS= c_typecast.cpp ansi_c_parser.cpp \
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
      type2name.cpp fix_symbol.cpp printf_formatter.cpp \

OBJS= $(OBJDIR)/y.tab.o $(OBJDIR)/lex.yy.o
HOSTOBJS= $(HOSTOBJDIR)/y.tab.o $(HOSTOBJDIR)/lex.yy.o

INCLUDES= -I .. -I ../util

all: $(OBJDIR)/.deps
	$(MAKE) -C cpp
	$(MAKE) -C headers
	$(MAKE) $(OBJDIR)/ansi-c.o

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

include $(ESBMCDIR)/common

###############################################################################

C2GOTO_OBJS= $(HOSTOBJS)

C2GOTO_DEPS= ../util/$(HOSTOBJDIR)/util.o ../langapi/$(HOSTOBJDIR)/langapi.o \
	     ../big-int/$(HOSTOBJDIR)/bigint.o \
	     ../goto-programs/$(HOSTOBJDIR)/goto-programs.o \
	     ../pointer-analysis/$(HOSTOBJDIR)/pointer-analysis.o \
	     cpp/$(HOSTOBJDIR)/preproc.o headers/$(HOSTOBJDIR)/headers.o

$(HOSTOBJDIR)/c2goto: c2goto.cpp $(HOSTOBJDIR)/cprover_blank_library.o | $(C2GOTO_OBJS)
	$(HOSTFINLINKBIN) $(HOSTCXXFLAGS) $^ $(HOSTFINLINKOUT) $@ $(C2GOTO_OBJS) $(C2GOTO_DEPS) $(HOSTFINLINKTAIL)

$(OBJDIR)/clib32.o: $(HOSTOBJDIR)/c2goto library/*.c
	./$(HOSTOBJDIR)/c2goto library/*.c $(CLIB32FLAG) --output clib32.goto --no-lock-check $(C2GOTO_INCS)
	./headers/flail.sh clib32_buf clib32.goto | $(CXX) -x c - -c -o $@

$(OBJDIR)/clib64.o: $(HOSTOBJDIR)/c2goto library/*.c
	./$(HOSTOBJDIR)/c2goto library/*.c $(CLIB64FLAG) --output clib64.goto --no-lock-check $(C2GOTO_INCS)
	./headers/flail.sh clib64_buf clib64.goto | $(CXX) -x c - -c -o $@

$(OBJDIR)/clib.o: $(HOSTOBJDIR)/c2goto $(OBJDIR)/clib32.o $(OBJDIR)/clib64.o
	$(PARTLINKLD) $(OBJDIR)/clib32.o $(OBJDIR)/clib64.o $(PARTLINKOUT) $(OBJDIR)/clib.o $(PARTLINKOPTS)

###############################################################################

$(OBJDIR)/ansi-c.o: $(OBJS) $(OBJDIR)/cprover_library.o $(OBJDIR)/clib.o cpp/$(OBJDIR)/preproc.o headers/$(OBJDIR)/headers.o
	$(PARTLINKLD) $(PARTLINKOUT) $@ $^ $(PARTLINKOPTS)

clean:
	$(MAKE) -C cpp clean
	$(MAKE) -C headers clean
	-rm -f $(OBJDIR)/ansi-c.o
	-rm -f $(OBJS) y.tab.h y.tab.cpp lex.yy.cpp y.tab.cpp.output y.output
	-rm -f $(OBJDIR)/c2goto
	-rm -f *.goto
	-rm -f $(OBJDIR)/clib32.o $(OBJDIR)/clib64.o $(OBJDIR)/clib.o
	-rm -f $(OBJDIR)/cprover_blank_library.o $(OBJDIR)/cprover_library.o
	-rm $(OBJDIR)/.deps
