if(NOT ENABLE_SMTLIB)
  set(ESBMC_ENABLE_smtlib 0 PARENT_SCOPE)
  return()
endif()

find_package(BISON 2.6.1 REQUIRED)
find_package(FLEX 2.6.1 REQUIRED)

BISON_TARGET(smtliby smtlib.ypp ${CMAKE_CURRENT_BINARY_DIR}/smtlib.cpp COMPILE_FLAGS "-d -psmtlib")
FLEX_TARGET(smtlibl smtlib_tok.lpp ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp COMPILE_FLAGS "--header-file=smtlib_tok.hpp -Psmtlib_tok -osmtlib_tok.cpp ${OS_FLEX_SMTLIB_FLAGS}")
ADD_FLEX_BISON_DEPENDENCY(smtlibl smtliby)

# For Coverage
configure_file(smtlib.ypp  ${CMAKE_BINARY_DIR}/smtlib.ypp COPYONLY)
configure_file(smtlib_tok.lpp  ${CMAKE_BINARY_DIR}/smtlib_tok.lpp COPYONLY)

add_library(smtlib smtlib_conv.cpp ${CMAKE_CURRENT_BINARY_DIR}/smtlib.cpp ${CMAKE_CURRENT_BINARY_DIR}/lexer.cpp)
target_link_libraries(smtlib fmt::fmt)
target_include_directories(smtlib
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${Boost_INCLUDE_DIRS}
)

# The smtlib backend needs a larger-than-default stack size (at least in Debug
# builds) because the data structures built for the S-exprs in solver responses
# could be very large and the std::list destructors use a lot of stack space

# 20 MB ought to be enough for everyone
set(stacksize_dec 20971520)
set(stacksize_hex 0x1400000)

include(CheckLinkerFlag)

set(variants
    "/STACK:${stacksize_dec}"             # MSVC
    "-Wl,-stack_size,${stacksize_hex}"    # ld.lld, ld64 on macOS, see issue #1212
    "-Wl,-z,stack-size=${stacksize_dec}"  # ld.bfd, ld.gold
   )

if(NOT DEFINED has_stacksize_linker_flag)
  foreach(flag IN LISTS variants)
    unset(has_stacksize_linker_flag CACHE)
    check_linker_flag(CXX "${flag}" has_stacksize_linker_flag)
    if(has_stacksize_linker_flag)
      message(STATUS "Setting stack-size for executables to 20 MiB via '${flag}'")
      set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${flag}"
          CACHE STRING "Linker flags" FORCE)
      break()
    endif()
  endforeach()
endif()

if(NOT has_stacksize_linker_flag)
  message(WARNING
          "Unable to find flag to set stack-size for executables; "
          "option --smtlib might produce stack overflows for deeply nested "
          "terms; please let the ESBMC developers know about the compiler and "
          "linker used.")
endif()

# Add to solver link
target_link_libraries(solvers INTERFACE smtlib)

set(ESBMC_ENABLE_smtlib 1 PARENT_SCOPE)
set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} smtlib" PARENT_SCOPE)
