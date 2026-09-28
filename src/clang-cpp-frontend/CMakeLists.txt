add_subdirectory(abstract_includes)
add_library(clangcppfrontend clang_cpp_language.cpp
            clang_cpp_convert.cpp clang_cpp_adjust_expr.cpp clang_cpp_adjust_code.cpp
            clang_cpp_adjust_code_gen.cpp clang_cpp_convert_vft.cpp clang_cpp_convert_bind.cpp clang_cpp_main.cpp
            clang_cpp_adjust_irep2.cpp clang_cpp_exception_id.cpp clang_cpp_destructor_call.cpp)
target_include_directories(clangcppfrontend
    PRIVATE ${CLANG_INCLUDE_DIRS}
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
)

add_library(esbmc_internal_cpp esbmc_internal_cpp.cpp)
target_include_directories(esbmc_internal_cpp
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(esbmc_internal_cpp PUBLIC esbmc_common cpp_includes)

target_link_libraries(clangcppfrontend PUBLIC esbmc_common esbmc_internal_cpp fmt::fmt)
