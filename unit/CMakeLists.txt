message(STATUS "Testing enabled")
include(TestConfiguration)
add_subdirectory(testing-utils)

add_subdirectory(algorithms)
add_subdirectory(big-int)
#add_subdirectory(clang-c-frontend)
add_subdirectory(util)
add_subdirectory(c2goto)
