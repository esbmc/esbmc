
mangle(mangled_cpp "${CMAKE_SOURCE_DIR}/src/cpp/library"
    RECURSIVE 1
    PREFIX esbmc_libcpp_header_
    WILDCARD *
    MACRO ESBMC_FLAIL
    ACC_HEADERS_INTO ${CMAKE_CURRENT_BINARY_DIR}/cpp_includes.h
    LIST_DIRECTORIES FALSE)

add_library(cpp_includes ${mangled_cpp})
set_target_properties(cpp_includes PROPERTIES LINKER_LANGUAGE C
                                   PRIVATE_HEADER ${CMAKE_CURRENT_BINARY_DIR}/cpp_includes.h)

