
add_library(clangcppfrontend clang_cpp_language.cpp)
target_include_directories(clangcppfrontend 
    PRIVATE ${CLANG_INCLUDE_DIRS}
)