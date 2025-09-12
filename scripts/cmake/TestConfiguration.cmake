# This module configure and set the functions to make
# testing simpler


# UNIT TEST with catch2

Include(FetchContent)

FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v2.13.7)

FetchContent_MakeAvailable(Catch2)

list(APPEND CMAKE_MODULE_PATH
     ${catch2_SOURCE_DIR}/contrib  # Catch2 v2.x
     ${catch2_SOURCE_DIR}/extras   # Catch2 v3.x
)
include(Catch)
set(UNIT_TEST_LIB Catch2::Catch2)
if(EXISTS ${Catch2_SOURCE_DIR}/src/catch2/catch_all.hpp)
  file(CREATE_LINK ${Catch2_SOURCE_DIR}/src/catch2/catch_all.hpp
                   ${Catch2_SOURCE_DIR}/src/catch2/catch.hpp)
  set(UNIT_TEST_LIB ${UNIT_TEST_LIB} Catch2::Catch2WithMain)
endif()

# FUNCTIONS DEFINED

# Adds a new Unit based test
function (new_unit_test TARGET SRC LIBS)
  add_executable(${TARGET} ${SRC})
  target_include_directories(${TARGET} PRIVATE ${Boost_INCLUDE_DIRS})
  target_link_libraries(${TARGET} PRIVATE ${LIBS} ${UNIT_TEST_LIB} ${OS_INCLUDE_LIBS})
  catch_discover_tests(${TARGET})
endfunction()

# Add a new Fuzz based test
function (new_fuzz_test TARGET SRC LIBS)
  if(NOT ENABLE_FUZZER)
    return()
  endif()
  add_executable(${TARGET} ${SRC})
  add_test(NAME ${TARGET}-Fuzz COMMAND ${TARGET} -runs=6500000)
  target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANG_AND_ID:C,Clang>:-g -O1 -fsanitize=fuzzer>
                                           $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-g -O1 -fsanitize=fuzzer>)
  target_link_libraries(${TARGET} PRIVATE $<$<LINK_LANG_AND_ID:C,Clang>:-fsanitize=fuzzer> ${LIBS}
                                          $<$<LINK_LANG_AND_ID:CXX,Clang>:-fsanitize=fuzzer> ${LIBS})
endfunction()

# Add a new Fuzz based test (this will execute less runs)
function (new_fast_fuzz_test TARGET SRC LIBS)
  if(NOT ENABLE_FUZZER)
    return()
  endif()
  add_executable(${TARGET} ${SRC})
  add_test(NAME ${TARGET}-Fuzz COMMAND ${TARGET} -runs=1000)
  target_compile_options(${TARGET} PRIVATE $<$<COMPILE_LANG_AND_ID:C,Clang>:-g -O1 -fsanitize=fuzzer>
                                           $<$<COMPILE_LANG_AND_ID:CXX,Clang>:-g -O1 -fsanitize=fuzzer>)
  target_link_libraries(${TARGET} PRIVATE $<$<LINK_LANG_AND_ID:C,Clang>:-fsanitize=fuzzer> ${LIBS}
                                          $<$<LINK_LANG_AND_ID:CXX,Clang>:-fsanitize=fuzzer> ${LIBS})
endfunction()
