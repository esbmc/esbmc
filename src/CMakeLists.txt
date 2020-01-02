#################################
# Configuration                 #
#################################

# Adds custom modules from ESBMC and default Options
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/src/scripts/cmake/")

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
include(InstallFiles)
include(Sanitizers)


if(BUILD_STATIC)
  include(BuildStatic)
  add_definitions(-DBOOST_PYTHON_STATIC_LIB)
endif()

include(DefineLibM)

# This MUST be executed after BuildStatic since it sets Boost Static flags
find_package(Boost REQUIRED COMPONENTS filesystem date_time)
include(FindLLVM)

if(ENABLE_PYTHON)
  include(FindPython)
else()
  message(STATUS "Python support is off")
endif()

# Optimization
include(OptimizationCCache) 

#################################
# Modules                       #
#################################

include_directories(${CMAKE_SOURCE_DIR}/src)
add_subdirectory (goto-programs)
add_subdirectory (c2goto)
add_subdirectory (ansi-c)
add_subdirectory (clang-c-frontend)
add_subdirectory (cpp)
add_subdirectory (pointer-analysis)
add_subdirectory (langapi)
add_subdirectory (util)
add_subdirectory (big-int)
add_subdirectory (solvers)
add_subdirectory (clang-cpp-frontend)
add_subdirectory (goto-symex)
add_subdirectory (esbmc)

# Generate ac_config.h. This must be generated after solvers
configure_file (
 "${PROJECT_SOURCE_DIR}/src/scripts/cmake/cmake_config.in"
 "${PROJECT_BINARY_DIR}/src/ac_config.h"
)

# This should be added after all source files
include(Docs)