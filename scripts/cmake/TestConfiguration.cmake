# This module configure and set the functions to make
# testing simpler


# UNIT TEST with catch2

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.1)

FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(Catch)
set(UNIT_TEST_LIB Catch2::Catch2)

# Functions
function (new_unit_test TARGET SRC LIBS)
  add_executable(${TARGET} ${SRC})
  target_link_libraries(${TARGET} PRIVATE ${LIBS} ${UNIT_TEST_LIB})
  catch_discover_tests(${TARGET})
endfunction()

function (new_fuzz_test TARGET SRC LIBS)
  if(NOT ENABLE_FUZZER)
    return()
  endif()
  add_executable(${TARGET} ${SRC})
  add_test(NAME ${TARGET}-Fuzz COMMAND ${TARGET} -runs=6500000)
  target_compile_options(${TARGET} PRIVATE $<$<C_COMPILER_ID:Clang>:-g -O1 -fsanitize=fuzzer>)
  target_link_libraries(${TARGET} PRIVATE $<$<C_COMPILER_ID:Clang>:-fsanitize=fuzzer> ${LIBS})
endfunction()