# Activates ccache on build

include_guard(GLOBAL)

find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_FOUND}" CACHE STRING "")
  set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_FOUND}" CACHE STRING "")
  message(STATUS "Using ccache: ${CCACHE_FOUND}")
else()
  message(AUTHOR_WARNING "ccache not found, incremental builds will be slower")
endif()
