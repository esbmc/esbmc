# The include paths, definitions and Boost dependency shared by every ESBMC
# target, carried as usage requirements on one interface target.

include_guard(GLOBAL)

add_library(esbmc_common INTERFACE)
add_library(ESBMC::common ALIAS esbmc_common)
target_include_directories(esbmc_common INTERFACE
  "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>"
  "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>")

target_compile_definitions(esbmc_common INTERFACE
  BOOST_ALL_NO_LIB
  YAML_CPP_STATIC_DEFINE)

# Anything touching irep2 needs Boost's headers: irep2.h -> config.h ->
# cmdline.h -> boost/program_options.hpp. Boost_INCLUDE_DIRS is named alongside
# Boost::headers deliberately -- a target that has only the imported target and
# not the path still builds on Linux, where Boost lives on an implicit search
# path, and fails on macOS/Homebrew, where it does not.
if(NOT Boost_INCLUDE_DIRS)
  message(FATAL_ERROR
    "Boost was found but Boost_INCLUDE_DIRS is empty, so ESBMC cannot put its "
    "headers on the include path. Check the Boost package at ${Boost_DIR}.")
endif()
target_link_libraries(esbmc_common INTERFACE Boost::headers)
target_include_directories(esbmc_common INTERFACE
  "$<BUILD_INTERFACE:${Boost_INCLUDE_DIRS}>")

add_library(esbmc_boost INTERFACE)
add_library(ESBMC::boost ALIAS esbmc_boost)
target_link_libraries(esbmc_boost INTERFACE
  Boost::date_time Boost::program_options Boost::iostreams)
foreach(component filesystem system)
  if(TARGET Boost::${component})
    target_link_libraries(esbmc_boost INTERFACE Boost::${component})
  endif()
endforeach()

# Enabled frontends, consumed as #ifdef by src/esbmc/globals.cpp,
# src/c2goto/cprover_library.cpp and others.
foreach(frontend SOLIDITY JIMPLE PYTHON LD)
  if(ENABLE_${frontend}_FRONTEND)
    target_compile_definitions(esbmc_common INTERFACE ENABLE_${frontend}_FRONTEND)
  endif()
endforeach()

if(ENABLE_GOTO_CONTRACTOR)
  target_compile_definitions(esbmc_common INTERFACE ENABLE_GOTO_CONTRACTOR)
endif()

# Debug aid: prove every simplifier rewrite equivalent with an SMT solver
if(ENABLE_SIMPLIFIER_EQUIVALENCE_CHECK)
  target_compile_definitions(esbmc_common INTERFACE ENABLE_SIMPLIFIER_EQUIVALENCE_CHECK)
endif()

# Adds the ESBMC:: alias other projects link by, so a superbuild consuming ESBMC
# through add_subdirectory()/FetchContent names targets the same way an
# installed package would.
function(esbmc_add_alias target)
  if(TARGET ${target} AND NOT TARGET ESBMC::${target})
    add_library(ESBMC::${target} ALIAS ${target})
  endif()
endfunction()
