add_library(clangcfrontendast build_ast.cpp vfs_adapter.cpp)
target_include_directories(clangcfrontendast
    PRIVATE ${CLANG_INCLUDE_DIRS}
)
set_target_properties(clangcfrontendast PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(clangcfrontendast filesystem ${ESBMC_CLANG_LIBS})
