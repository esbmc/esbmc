#################################
# Configuration                 #
#################################

# Adds custom modules from ESBMC and default Options
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/src/scripts/cmake/")

include(Options)
include(SendFileHack)

# Generate ac_config.h
configure_file (
 "${PROJECT_SOURCE_DIR}/src/scripts/cmake/cmake_config.in"
 "${PROJECT_BINARY_DIR}/src/ac_config.h"
)

if(BUILD_STATIC)
    set(Boost_USE_STATIC_LIBS ON)
    set(Boost_USE_STATIC_RUNTIME ON)
endif()

# This MUST be executed after BuildStatic since it sets Boost Static flags
find_package(Boost COMPONENTS filesystem date_time)
include(FindLLVM)

if(ENABLE_PYTHON)
  add_definitions(-DBOOST_PYTHON_STATIC_LIB)
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
add_subdirectory (big-int)
add_subdirectory (util)
add_subdirectory (langapi)
add_subdirectory (solvers)
add_subdirectory (goto-symex)
add_subdirectory (goto-programs)
add_subdirectory (pointer-analysis)
add_subdirectory (ansi-c)
add_subdirectory (cpp)
add_subdirectory (clang-c-frontend)
add_subdirectory (clang-cpp-frontend)
add_subdirectory (c2goto)
add_subdirectory (esbmc)
