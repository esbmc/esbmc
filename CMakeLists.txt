#################################
# Project Information           #
#################################
cmake_minimum_required (VERSION 3.7)
project (ESBMC)

set (ESBMC_VERSION_MAJOR 6)
set (ESBMC_VERSION_MINOR 4)
set (ESBMC_VERSION_PATCH 0)
set (ESBMC_VERSION "${ESBMC_VERSION_MAJOR}.${ESBMC_VERSION_MINOR}.${ESBMC_VERSION_PATCH}")

# The only default solver available is smtlib
set (ESBMC_AVAILABLE_SOLVERS "smtlib")

#################################
# Configuration                 #
#################################

# Adds custom modules from ESBMC and default Options
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/scripts/cmake/")

# Set a default build type if none was specified
set(default_build_type "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo" "Sanitizer")
endif()

include(Options)
include(SendFileHack)
include(AppleConfiguration)
include(WindowsConfiguration)
include(InstallFiles)
include(Sanitizers)
include(ClangTidy)
include(Coverage)

if(BUILD_STATIC)
  include(BuildStatic)
endif()

include(DefineLibM)

# Python
find_package(Python)

# This MUST be executed after BuildStatic since it sets Boost Static flags
find_package(Boost REQUIRED COMPONENTS filesystem system date_time)
include(FindLLVM)

# Optimization
include(OptimizationCCache)

add_subdirectory(src)

# Generate ac_config.h. This must be generated after solvers
configure_file (
 "${PROJECT_SOURCE_DIR}/scripts/cmake/cmake_config.in"
 "${PROJECT_BINARY_DIR}/src/ac_config.h"
)

# This should be added after all source files
include(Docs)

include(CTest)
if(BUILD_TESTING)
    enable_testing()
    include_directories(src)
    add_subdirectory(unit)
endif()
if(ENABLE_REGRESSION)
    add_subdirectory(regression)
endif()
