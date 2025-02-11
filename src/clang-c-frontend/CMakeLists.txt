add_subdirectory(headers)
add_subdirectory(AST)

add_library(clangcfrontend_stuff clang_c_language.cpp clang_c_convert.cpp
            clang_c_main.cpp clang_c_adjust_expr.cpp typecast.cpp clang_c_adjust_code.cpp
            clang_c_convert_literals.cpp clang_headers.cpp padding.cpp symbolic_types.cpp
            clang_c_adjust_polymorphic_functions.cpp)
target_include_directories(clangcfrontend_stuff
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CLANG_INCLUDE_DIRS}
    PRIVATE ${Boost_INCLUDE_DIRS}
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
)
target_link_libraries(clangcfrontend_stuff gotoprograms ${cheaders_lib} ${ESBMC_CLANG_LIBS})

add_library(clangcfrontend INTERFACE)
target_link_libraries(clangcfrontend INTERFACE clangcfrontend_stuff clangcfrontendast)
