# The neurosym backend drives an external NeuroSym Python program at runtime;
# there is no library to find or link. It reuses the smtlib backend's
# serializer and response parser, so it cannot be built without it. As
# ENABLE_NEUROSYM defaults to ON, degrade gracefully instead of failing the
# configure when smtlib is unavailable (e.g. on Windows CI).
if(NOT ENABLE_NEUROSYM OR NOT ENABLE_SMTLIB)
  if(ENABLE_NEUROSYM AND NOT ENABLE_SMTLIB)
    message(STATUS "Disabling the neurosym backend: it requires ENABLE_SMTLIB")
  endif()
  set(ESBMC_ENABLE_neurosym 0 PARENT_SCOPE)
  return()
endif()

add_library(solverneurosym neurosym_conv.cpp)
target_link_libraries(solverneurosym smtlib fmt::fmt)
target_include_directories(solverneurosym
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_BINARY_DIR}/src
    PRIVATE ${Boost_INCLUDE_DIRS}
)

target_link_libraries(solvers INTERFACE solverneurosym)

set(ESBMC_ENABLE_neurosym 1 PARENT_SCOPE)
set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} neurosym" PARENT_SCOPE)
