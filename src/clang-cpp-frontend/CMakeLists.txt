
add_library(clangcppfrontend clang_cpp_language.cpp expr2cpp.cpp)
target_include_directories(clangcppfrontend 
    PRIVATE ${CLANG_INCLUDE_DIRS}
    PRIVATE ${Boost_INCLUDE_DIRS}
)
