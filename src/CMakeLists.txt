cmake_minimum_required (VERSION 3.7)
project (ESBMC)

set (ESBMC_VERSION_MAJOR 6)
set (ESBMC_VERSION_MINOR 0)
set (ESBMC_VERSION_PATCH 0)
set (ESBMC_VERSION "${ESBMC_VERSION_MAJOR}.${ESBMC_VERSION_MINOR}.${ESBMC_VERSION_PATCH}")

message("ESBMC ${ESBMC_VERSION} cmake started!")

# Adds custom modules from ESBMC and default Options
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/scripts/cmake/")
include(Options)

# The only default solver available is smtlib
set (ESBMC_AVAILABLE_SOLVERS "smtlib")
set (ESBMC_SOLVERS 4.6.0)

# Configuration
include(SendFileHack)
configure_file (
 "${PROJECT_SOURCE_DIR}/scripts/cmake/cmake_config.in"
 "${PROJECT_BINARY_DIR}/ac_config.h"
)
 #include(DefineLibm)

#if(ENABLE_WERROR)
#  include(CompilerWerror)
#endif()

#if(BUILD_STATIC)
#  include(BuildStatic)
#else()
#  message(STATUS "ESBMC will be built in dynamic mode")
#endif()

# This MUST be executed after BuildStatic since it sets Boost Static flags
include(FindBoost) # This is a CMake default module
#include(FindLLVM)

#if(ENABLE_PYTHON)
#  add_definitions(-DBOOST_PYTHON_STATIC_LIB)
#  include(FindPython)
#else()
#  message(STATUS "Python support is off")
#endif()

include_directories (${CMAKE_SOURCE_DIR})
#include_directories (${CMAKE_BINARY_DIR})

# Optimization
#include(OptimizationCCache) 

add_subdirectory (big-int)
add_subdirectory (util)
add_subdirectory (langapi)
add_subdirectory (solvers)
#add_subdirectory (goto-symex)
#add_subdirectory (goto-programs)
#add_subdirectory (pointer-analysis)
#add_subdirectory (ansi-c)
#add_subdirectory (cpp)
#add_subdirectory (clang-c-frontend)
#add_subdirectory (clang-cpp-frontend)
#add_subdirectory (c2goto)
#add_subdirectory (esbmc)
