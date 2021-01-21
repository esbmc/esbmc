message(STATUS "Testing enabled")
find_package(Boost COMPONENTS unit_test_framework REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})

add_subdirectory(big-int)
add_subdirectory(clang-c-frontend)
add_subdirectory(c2goto)