add_subdirectory(cpp)

find_package(BISON)
find_package(FLEX)

BISON_TARGET(cparser parser.ypp ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp COMPILE_FLAGS "-d -pyyansi_c")
FLEX_TARGET(cscanner scanner.lpp ${CMAKE_CURRENT_BINARY_DIR}/scanner.cpp COMPILE_FLAGS "--header-file=scanner.hpp -Pyyansi_c")
ADD_FLEX_BISON_DEPENDENCY(cscanner cparser)

add_library(ansicfrontend_stuff ${CMAKE_CURRENT_BINARY_DIR}/parser.cpp ${CMAKE_CURRENT_BINARY_DIR}/scanner.cpp ansi_c_parser.cpp ansi_c_language.cpp c_main.cpp c_final.cpp trans_unit.cpp ansi_c_typecheck.cpp c_preprocess.cpp type2name.cpp c_typecheck_base.cpp c_typecheck_initializer.cpp c_typecheck_typecast.cpp c_typecheck_code.cpp c_typecheck_expr.cpp c_typecheck_type.cpp unescape_string.cpp parse_float.cpp convert_float_literal.cpp convert_integer_literal.cpp c_typecheck_argc_argv.cpp ansi_c_parse_tree.cpp preprocessor_line.cpp convert_character_literal.cpp convert_string_literal.cpp ansi_c_convert.cpp ansi_c_convert_type.cpp)
target_include_directories(ansicfrontend_stuff
    PRIVATE ${Boost_INCLUDE_DIRS}
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(ansicfrontend_stuff)

add_library(ansicfrontend INTERFACE)
target_link_libraries(ansicfrontend INTERFACE ansicfrontend_stuff ansiccpp)