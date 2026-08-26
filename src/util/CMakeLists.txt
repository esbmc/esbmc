add_library(algorithms ssa/algorithms.cpp)
target_include_directories(algorithms
    PRIVATE ${Boost_INCLUDE_DIRS}
    )

add_library(cache ssa/cache.cpp ssa/fingerprint.cpp ssa/proof_cache.cpp)
# fingerprint.cpp digests SSA steps, so cache now depends on irep2 and the
# util core. Declared here rather than at each consumer: without it the Linux
# linker, which does not revisit an archive, leaves the irep2 vtables and
# irept::remove_ref undefined in anything that links cache.
target_link_libraries(cache algorithms irep2 util_esbmc)

add_library(filesystem base/filesystem.cpp)
target_include_directories(filesystem
    PRIVATE ${Boost_INCLUDE_DIRS}
    )
target_link_libraries(filesystem PUBLIC ${Boost_LIBRARIES})

add_library(string_pool base/string_pool.cpp)

add_library(usr_utils base/usr_utils.cpp)

add_library(util_esbmc irep/xml_irep.cpp base/xml.cpp
        arith/arith_tools.cpp expr/base_type.cpp config/cmdline.cpp config/config.cpp config/config_file.cpp symtab/context.cpp
        base/cwe_mapping.cpp
        lang/fallible_calls.cpp
        expr/expr_util.cpp base/i2string.cpp irep/location.cpp
        arith/mp_arith.cpp symtab/namespace.cpp config/parseoptions.cpp symtab/rename.cpp
        base/threeval.cpp lang/typecheck.cpp arith/bitvector.cpp lang/parser.cpp symtab/replace_symbol.cpp
        base/string_pool.cpp config/options.cpp lang/c_misc.cpp
        expr/expr_simplifier.cpp expr/expr_reassociate.cpp
        expr/string2array.cpp expr/array2string.cpp base/time_stopping.cpp symtab/symbol.cpp
        symtab/symbol_generator.cpp expr/symbolic_types.cpp
        expr/array_name.cpp base/union_find.cpp
        irep/std_types.cpp irep/std_code.cpp arith/format_constant.cpp
        irep/irep_serialization.cpp symtab/symbol_serialization.cpp arith/fixedbv.cpp
        base/signal_catcher.cpp irep/migrate.cpp symtab/show_symbol_table.cpp
        expr/type_byte_size.cpp ssa/goto_expr_factory.cpp
        expr/string_constant.cpp lang/c_types.cpp lang/python_types.cpp arith/ieee_float.cpp lang/c_qualifiers.cpp
        lang/c_sizeof.cpp lang/c_link.cpp lang/c_typecast.cpp symtab/fix_symbol.cpp
        lang/c_expr2string.cpp lang/cpp_expr2string.cpp expr/type2name.cpp
        lang/exception_specification.cpp
        message/message.cpp base/encoding.cpp irep/expr.cpp irep/irep.cpp irep/type.cpp base/yaml_parser.cpp
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
