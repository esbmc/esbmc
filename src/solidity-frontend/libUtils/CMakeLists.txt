file(GLOB sources "*.cpp")
file(GLOB headers "*.hpp")

set(JSON_BuildTests OFF CACHE INTERNAL "")

add_subdirectory(nlohmann_json)

add_library(solidityutil ${sources} ${headers})

target_include_directories(solidityutil
    PRIVATE ${CLANG_INCLUDE_DIRS}
    PRIVATE "${CMAKE_SOURCE_DIR}"
)
set_target_properties(solidityutil PROPERTIES COMPILE_FLAGS "-fno-rtti")
