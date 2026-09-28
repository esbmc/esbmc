# Precompiled headers for the irep2 core.
#
# irep2.h and irep2_expr.h are the heaviest headers in the tree and are pulled
# in by most of the C++ sources, so they are precompiled once and reused. The
# reusing targets must share the PCH-relevant compile flags, which is why
# clangcfrontendast (built -fno-rtti) is not in the list.

include_guard(GLOBAL)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "10.0.0")
  message(AUTHOR_WARNING "GCC-9 throws a Segmentation fault. Skipping precompiled headers.")
  return()
endif()

target_precompile_headers(util_esbmc PRIVATE
  ${CMAKE_SOURCE_DIR}/src/irep2/irep2.h
  ${CMAKE_SOURCE_DIR}/src/irep2/irep2_expr.h)

set(esbmc_pch_targets
  clangcfrontend_stuff
  clangcppfrontend
  gotoprograms
  symex
  pointeranalysis
  gotoalgorithms
  abstract-interpretation
  smt
  solve)
# esbmc-driver is deliberately absent: with the PCH force-included, GCC 13 at
# -O2 reports a false -Wstringop-overflow on the ESBMC_AVAILABLE_SOLVERS
# concatenation in driver.cpp, which -Werror turns into a build failure.

foreach(target IN LISTS esbmc_pch_targets)
  if(TARGET ${target})
    target_precompile_headers(${target} REUSE_FROM util_esbmc)
  endif()
endforeach()
