if(NOT ENABLE_BITWUZLLOB)
  set(ESBMC_ENABLE_bitwuzllob 0 PARENT_SCOPE)
  return()
endif()

# The bitwuzllob backend drives an external Mallob binary at runtime; there is
# no library to find or link. It reuses the smtlib backend's serializer and
# response parser, so it cannot be built without it.
if(NOT ENABLE_SMTLIB)
  message(FATAL_ERROR "ENABLE_BITWUZLLOB requires ENABLE_SMTLIB")
endif()

add_library(solverbitwuzllob bitwuzllob_conv.cpp)
target_link_libraries(solverbitwuzllob smtlib fmt::fmt)
target_include_directories(solverbitwuzllob
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${Boost_INCLUDE_DIRS}
)

target_link_libraries(solvers INTERFACE solverbitwuzllob)

set(ESBMC_ENABLE_bitwuzllob 1 PARENT_SCOPE)
set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} bitwuzllob" PARENT_SCOPE)
