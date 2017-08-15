SUBDIRS = library
noinst_LTLIBRARIES = libcxxscanner.la libcxx.la

AM_CXXFLAGS = $(ESBMC_CXXFLAGS) -I$(top_srcdir)
AM_YFLAGS = -d -pyycpp
AM_LFLAGS = --header-file=scanner.hpp -Pyycpp -o scanner.cpp

libcxxscanner_la_SOURCES = scanner.lpp
libcxxscanner_la_CXXFLAGS = -Wno-error $(ESBMC_CXXFLAGS_WARNINGS) \
      -I$(top_srcdir)

libcxx_la_SOURCES = cpp_id.cpp cpp_language.cpp expr2cpp.cpp cpp_parser.cpp \
      cpp_typecheck.cpp cpp_convert_type.cpp \
      cpp_typecheck_expr.cpp cpp_typecheck_code.cpp \
      cpp_typecheck_type.cpp parse.cpp cpp_parse_tree.cpp \
      cpp_token_buffer.cpp cpp_typecheck_fargs.cpp \
      cpp_typecheck_resolve.cpp cpp_util.cpp \
      cpp_typecheck_function.cpp cpp_typecheck_namespace.cpp \
      cpp_name.cpp cpp_is_pod.cpp cpp_scope.cpp \
      cpp_typecheck_find_constructor.cpp template_map.cpp \
      cpp_scopes.cpp cpp_typecheck_linkage_spec.cpp \
      cpp_declarator.cpp cpp_type2name.cpp cpp_instantiate_template.cpp \
      cpp_typecheck_template.cpp cpp_typecheck_function_bodies.cpp \
      cpp_typecheck_initializer.cpp cpp_typecheck_compound_type.cpp \
      cpp_constructor.cpp cpp_destructor.cpp \
      cpp_final.cpp cpp_typecheck_conversions.cpp\
      cpp_typecheck_declaration.cpp cpp_declarator_converter.cpp \
      cpp_declaration.cpp cpp_namespace_spec.cpp \
      cpp_typecheck_using.cpp cpp_exception_id.cpp \
      cpp_typecheck_enum_type.cpp cpp_typecheck_bases.cpp \
      cpp_typecheck_constructor.cpp cpp_typecheck_virtual_table.cpp
libcxx_la_LIBADD = libcxxscanner.la

cppincludedir = $(includedir)/cpp
cppinclude_HEADERS = cpp_class_type.h cpp_convert_type.h cpp_declaration.h \
      cpp_declarator.h cpp_declarator_converter.h cpp_enum_type.h \
      cpp_exception_id.h cpp_final.h cpp_id.h cpp_item.h cpp_language.h \
      cpp_linkage_spec.h cpp_member_spec.h cpp_name.h cpp_namespace_spec.h \
      cpp_parse_tree.h cpp_parser.h cpp_scope.h cpp_scopes.h \
      cpp_storage_spec.h cpp_template_args.h cpp_template_type.h \
      cpp_token.h cpp_token_buffer.h cpp_type2name.h cpp_typecast.h \
      cpp_typecheck.h cpp_typecheck_fargs.h cpp_typecheck_resolve.h \
      cpp_using.h cpp_util.h expr2cpp.h template_map.h tokens.h

clean-local:
	-rm scanner.cpp scanner.hpp
.PHONY: clean-local
