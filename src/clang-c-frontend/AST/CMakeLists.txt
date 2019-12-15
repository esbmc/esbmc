add_library(clangcfrontendast build_ast.cpp)
target_include_directories(clangcfrontendast 
    PRIVATE ${CLANG_INCLUDE_DIRS}
)
set_target_properties(clangcfrontendast PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(clangcfrontendast clangTooling clangAST clangIndex)