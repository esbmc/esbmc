add_library(algorithms ssa/algorithms.cpp)
target_include_directories(algorithms
    PRIVATE ${Boost_INCLUDE_DIRS}
    )

add_library(cache ssa/cache.cpp)
target_link_libraries(cache algorithms)

add_library(filesystem base/filesystem.cpp)
target_include_directories(filesystem
    PRIVATE ${Boost_INCLUDE_DIRS}
    )
target_link_libraries(filesystem PUBLIC ${Boost_LIBRARIES})

add_library(string_pool base/string_pool.cpp)

add_library(usr_utils base/usr_utils.cpp)

add_library(util_esbmc xml_irep.cpp base/xml.cpp
        arith_tools.cpp base_type.cpp config/cmdline.cpp config/config.cpp config/config_file.cpp symtab/context.cpp
        base/cwe_mapping.cpp
        fallible_calls.cpp
        expr_util.cpp base/i2string.cpp location.cpp
        mp_arith.cpp symtab/namespace.cpp config/parseoptions.cpp symtab/rename.cpp
        base/threeval.cpp typecheck.cpp bitvector.cpp parser.cpp symtab/replace_symbol.cpp
        base/string_pool.cpp config/options.cpp c_misc.cpp
        expr_simplifier.cpp expr_reassociate.cpp
        string2array.cpp array2string.cpp base/time_stopping.cpp symtab/symbol.cpp
        symtab/symbol_generator.cpp symbolic_types.cpp
        array_name.cpp base/union_find.cpp
        std_types.cpp std_code.cpp format_constant.cpp
        irep_serialization.cpp symtab/symbol_serialization.cpp fixedbv.cpp
        base/signal_catcher.cpp migrate.cpp symtab/show_symbol_table.cpp
        type_byte_size.cpp ssa/goto_expr_factory.cpp
        string_constant.cpp c_types.cpp python_types.cpp ieee_float.cpp c_qualifiers.cpp
        c_sizeof.cpp c_link.cpp c_typecast.cpp symtab/fix_symbol.cpp
        c_expr2string.cpp cpp_expr2string.cpp type2name.cpp
        exception_specification.cpp
        message/message.cpp base/encoding.cpp expr.cpp irep.cpp type.cpp base/yaml_parser.cpp
        )

# Boost is needed by anything that touches irep2
target_include_directories(util_esbmc
        PUBLIC ${CMAKE_BINARY_DIR}/src
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
        PUBLIC ${Boost_INCLUDE_DIRS}
        lib
        )

target_compile_definitions(util_esbmc PUBLIC BOOST_ALL_NO_LIB YAML_CPP_STATIC_DEFINE)
target_link_libraries(util_esbmc PUBLIC irep2 fmt::fmt string_pool yaml-cpp::yaml-cpp ${Boost_LIBRARIES} PRIVATE langapi)

target_link_libraries(algorithms PUBLIC gotoprograms)
