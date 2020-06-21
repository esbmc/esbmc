find_package(BISON)
find_package(FLEX)

BISON_TARGET(cpy cpy.y ${CMAKE_CURRENT_BINARY_DIR}/cpy.c COMPILE_FLAGS "-pcpp -d --defines=cpy.h" DEFINES_FILE ${CMAKE_CURRENT_BINARY_DIR}/cpy.h)
# For Coverage
configure_file(cpy.y  ${CMAKE_BINARY_DIR}/cpy.y COPYONLY)

add_library(ansiccpp ${CMAKE_CURRENT_BINARY_DIR}/cpy.c cpp.c compat.c hooks.c token.c)
target_include_directories(ansiccpp BEFORE
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ansiccpp c2gotoheaders)