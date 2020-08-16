add_library(clangcfrontendast build_ast.cpp)
target_include_directories(clangcfrontendast 
    PRIVATE ${CLANG_INCLUDE_DIRS}
)
if (NOT WIN32)
set_target_properties(clangcfrontendast PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()
target_link_libraries(clangcfrontendast clangTooling clangAST clangIndex)