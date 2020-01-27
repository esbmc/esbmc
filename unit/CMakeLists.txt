message(STATUS "Testing enabled")
find_package(Boost COMPONENTS unit_test_framework REQUIRED)

include_directories(${Boost_INCLUDE_DIRS})
if(NOT BUILD_STATIC)
  add_definitions(-DBOOST_TEST_DYN_LINK)
endif()
add_subdirectory(big-int)
add_subdirectory(util)

