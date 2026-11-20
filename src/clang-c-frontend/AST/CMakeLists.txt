add_library(clangcfrontendast build_ast.cpp vfs_adapter.cpp)
target_include_directories(clangcfrontendast
    PRIVATE ${CLANG_INCLUDE_DIRS}
)
set_target_properties(clangcfrontendast PROPERTIES COMPILE_FLAGS "-fno-rtti")
target_link_libraries(clangcfrontendast PUBLIC esbmc_common filesystem ${ESBMC_CLANG_LIBS})

# MSVC 14.52 rejects clang's Diagnostic.h under C++23 (C2338, #10885).
if(MSVC)
  set_target_properties(clangcfrontendast PROPERTIES CXX_STANDARD 20)
endif()
