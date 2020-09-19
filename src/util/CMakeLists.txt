add_library(util_esbmc irep2.cpp
    arith_tools.cpp base_type.cpp cmdline.cpp config.cpp context.cpp
    expr_util.cpp i2string.cpp location.cpp message.cpp
    language_file.cpp mp_arith.cpp namespace.cpp parseoptions.cpp rename.cpp
    threeval.cpp typecheck.cpp bitvector.cpp parser.cpp replace_symbol.cpp
    string_container.cpp options.cpp c_misc.cpp
    simplify_expr.cpp dstring.cpp simplify_expr2.cpp ui_message.cpp
    simplify_utils.cpp string2array.cpp time_stopping.cpp symbol.cpp
    type_eq.cpp guard.cpp array_name.cpp message_stream.cpp union_find.cpp
    xml.cpp xml_irep.cpp std_types.cpp std_code.cpp format_constant.cpp
    irep_serialization.cpp symbol_serialization.cpp fixedbv.cpp
    signal_catcher.cpp migrate.cpp show_symbol_table.cpp
    crypto_hash.cpp type_byte_size.cpp
    string_constant.cpp c_types.cpp ieee_float.cpp c_qualifiers.cpp
    c_sizeof.cpp c_link.cpp c_typecast.cpp fix_symbol.cpp
)

# Boost is needed by anything that touches irep2
target_include_directories(util_esbmc
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${Boost_INCLUDE_DIRS}
)

target_link_libraries(util_esbmc ${Boost_LIBRARIES})
