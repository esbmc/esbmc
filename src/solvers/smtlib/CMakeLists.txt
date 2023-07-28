
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
set(stacksize 20971520)  # 20 MB ought to be enough for everyone
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /STACK:${stacksize}" CACHE STRING "Linker flags" FORCE)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
  set(stacksize 0x1400000) # macOS uses hex by default (see Issue #1212)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-stack_size,${stacksize}" CACHE STRING "Linker flags" FORCE)
else()
  # GNU probably
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-z,stack-size=${stacksize}" CACHE STRING "Linker flags" FORCE)
endif()
