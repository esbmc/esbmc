add_library(clangcppfrontend clang_cpp_language.cpp
            clang_cpp_convert.cpp clang_cpp_adjust_expr.cpp clang_cpp_adjust_code.cpp expr2cpp.cpp)
target_precompile_headers(clangcppfrontend PRIVATE ${CMAKE_SOURCE_DIR}/src/util/irep2.h)
target_include_directories(clangcppfrontend
    PRIVATE ${CLANG_INCLUDE_DIRS}
    PRIVATE ${Boost_INCLUDE_DIRS}
)
