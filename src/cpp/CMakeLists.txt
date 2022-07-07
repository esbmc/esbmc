
FLEX_TARGET(cppscanner scanner.lpp ${CMAKE_CURRENT_BINARY_DIR}/scanner.cpp COMPILE_FLAGS "--header-file=scanner.hpp -Pyycpp")

add_library(cppfrontend ${CMAKE_CURRENT_BINARY_DIR}/scanner.cpp cpp_id.cpp cpp_language.cpp cpp_parser.cpp cpp_typecheck.cpp cpp_convert_type.cpp cpp_typecheck_expr.cpp cpp_typecheck_code.cpp cpp_typecheck_type.cpp parse.cpp cpp_parse_tree.cpp cpp_token_buffer.cpp cpp_typecheck_fargs.cpp cpp_typecheck_resolve.cpp cpp_util.cpp cpp_typecheck_function.cpp cpp_typecheck_namespace.cpp cpp_name.cpp cpp_is_pod.cpp cpp_scope.cpp template_map.cpp cpp_scopes.cpp cpp_typecheck_linkage_spec.cpp cpp_declarator.cpp cpp_type2name.cpp cpp_instantiate_template.cpp cpp_typecheck_template.cpp cpp_typecheck_function_bodies.cpp cpp_typecheck_initializer.cpp cpp_typecheck_compound_type.cpp cpp_constructor.cpp cpp_destructor.cpp cpp_final.cpp cpp_typecheck_conversions.cpp cpp_typecheck_declaration.cpp cpp_declarator_converter.cpp cpp_declaration.cpp cpp_namespace_spec.cpp cpp_typecheck_using.cpp cpp_exception_id.cpp cpp_typecheck_enum_type.cpp cpp_typecheck_bases.cpp cpp_typecheck_constructor.cpp cpp_typecheck_virtual_table.cpp)
target_include_directories(cppfrontend
    PRIVATE ${Boost_INCLUDE_DIRS}
)
target_link_libraries(cppfrontend PUBLIC fmt::fmt)
