add_subdirectory(cpp)

find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)
BISON_TARGET(cparser parser.ypp ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp COMPILE_FLAGS "-d -pyyansi_c")
FLEX_TARGET(cscanner scanner.lpp ${CMAKE_CURRENT_BINARY_DIR}/scanner.cpp COMPILE_FLAGS "--header-file=scanner.hpp -Pyyansi_c")
ADD_FLEX_BISON_DEPENDENCY(cscanner cparser)

# For Coverage
configure_file(parser.ypp  ${CMAKE_BINARY_DIR}/parser.ypp COPYONLY)
configure_file(scanner.lpp  ${CMAKE_BINARY_DIR}/scanner.lpp COPYONLY)

# These libraries are separated since it gives a lot of register warnings.
add_library(ansic_with_register ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp ${CMAKE_CURRENT_BINARY_DIR}/scanner.cpp)
# target_compile_options(ansic_with_register PRIVATE -Wno-deprecated-register)
target_include_directories(ansic_with_register
    PRIVATE ${Boost_INCLUDE_DIRS}
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)


add_library(ansicfrontend_stuff ansi_c_parser.cpp ansi_c_language.cpp c_main.cpp c_final.cpp trans_unit.cpp ansi_c_typecheck.cpp c_preprocess.cpp type2name.cpp c_typecheck_base.cpp c_typecheck_initializer.cpp c_typecheck_typecast.cpp c_typecheck_code.cpp c_typecheck_expr.cpp c_typecheck_type.cpp unescape_string.cpp parse_float.cpp convert_float_literal.cpp convert_integer_literal.cpp c_typecheck_argc_argv.cpp ansi_c_parse_tree.cpp preprocessor_line.cpp convert_character_literal.cpp convert_string_literal.cpp ansi_c_convert.cpp ansi_c_convert_type.cpp)
target_include_directories(ansicfrontend_stuff
    PRIVATE ${Boost_INCLUDE_DIRS}
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(ansicfrontend_stuff ansic_with_register)

add_library(ansicfrontend INTERFACE)
target_link_libraries(ansicfrontend INTERFACE ansicfrontend_stuff ansiccpp)