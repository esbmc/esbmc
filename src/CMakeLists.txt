#################################
# Configuration                 #
#################################

# Adds custom modules from ESBMC and default Options
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/src/scripts/cmake/")

include(Options)
include(SendFileHack)
include(AppleConfiguration)

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