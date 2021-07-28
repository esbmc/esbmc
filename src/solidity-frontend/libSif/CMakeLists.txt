file(GLOB sources "*.cpp")
file(GLOB headers "*.hpp")

add_library(soliditysif ${sources} ${headers})
target_include_directories(soliditysif
    PRIVATE ${CLANG_INCLUDE_DIRS}
    PRIVATE "${CMAKE_SOURCE_DIR}"
    PRIVATE ../
)
target_link_libraries(soliditysif PRIVATE solidityutil nlohmann_json::nlohmann_json)
set_target_properties(soliditysif PROPERTIES COMPILE_FLAGS "-fno-rtti")
