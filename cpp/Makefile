ESBMCDIR= $(shell pwd)/..
include $(ESBMCDIR)/config.inc
include $(ESBMCDIR)/local.inc
MAINOBJ= $(OBJDIR)/cpp.o
all: .deps $(MAINOBJ)

SRCS= cpp_id.cpp cpp_language.cpp expr2cpp.cpp cpp_parser.cpp \
      cpp_typecheck.cpp cpp_convert_type.cpp \
      cpp_typecheck_expr.cpp cpp_typecheck_code.cpp \
      cpp_typecheck_type.cpp parse.cpp cpp_parse_tree.cpp \
      cpp_token_buffer.cpp cpp_typecheck_fargs.cpp \
      cpp_typecheck_resolve.cpp \
      cpp_typecheck_function.cpp cpp_typecheck_namespace.cpp \
      cpp_name.cpp cpp_is_pod.cpp cpp_scope.cpp \
      cpp_typecheck_find_constructor.cpp template_map.cpp \
      cpp_scopes.cpp \
      cpp_declarator.cpp \
      cpp_typecheck_template.cpp cpp_typecheck_function_bodies.cpp \
      cpp_typecheck_initializer.cpp cpp_typecheck_compound_type.cpp \
      cpp_constructor.cpp cpp_final.cpp cpp_typecheck_conversions.cpp\
      cpp_typecheck_declaration.cpp cpp_declarator_converter.cpp \
      cpp_declaration.cpp cpp_namespace_spec.cpp irep2name.cpp \
      cpp_typecheck_enum_type.cpp cpp_typecheck_bases.cpp \
      cpp_typecheck_constructor.cpp cpp_typecheck_virtual_table.cpp

GCCFLAGS += -D CPP_SYSTEMC_EXTENSION

# lex.yy.cpp isn't a source file we actually have for dependancy purposes,
# so instead put it in the object list.
OBJS= lex.yy.o

###############################################################################

lex.yy.cpp: scanner.l
	$(LEX) -Pyycpp -olex.yy.cpp scanner.l

$(OBJDIR)/lex.yy.o: lex.yy.cpp

clean:
	-rm -f $(OBJS) lex.yy.cpp y.tab.cpp.output .deps $(MAINOBJ)

include $(ESBMCDIR)/common
