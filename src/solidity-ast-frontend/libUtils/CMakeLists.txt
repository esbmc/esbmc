add_subdirectory(nlohmann_json)
add_library(solidityutil Utils.cpp)

target_include_directories(solidityutil
    PRIVATE ${CLANG_INCLUDE_DIRS}
)
set_target_properties(solidityutil PROPERTIES COMPILE_FLAGS "-fno-rtti")
#target_link_libraries(solidityutil clangTooling clangAST clangIndex)
