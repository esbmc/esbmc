# Module to list ESBMC Options

#############################
# ABOUT
#############################

#[[
This module sets all default options and variables with default values
to overwrite use the cmake cli, e.g -DENABLE_WERROR=On

Also, you can set some variables which are not defined directly here:
-DCMAKE_BUILD_TYPE which can be Release, Debug, RelWithDebInfo, etc (https://cmake.org/cmake/help/latest/variable/CMAKE_BUILD_TYPE.html)
-G which can be Ninja, Unix Makefile, Visual studio, etc...
]]

#############################
# GENERAL
#############################
option(ENABLE_WERROR "All warnings are treated as errors during compilation (default: OFF)" OFF)
option(ENABLE_PYTHON "Build esbmc with python support (default: OFF)" OFF)
option(BUILD_STATIC "Build ESBMC in static mode (default: OFF)" OFF)
option(BUILD_DOC "Build ESBMC documentation" OFF)
option(ENABLE_REGRESSION "Add Regressions Tests (default: OFF)" OFF)

#############################
# SOLVERS
#############################
option(ENABLE_BOOLECTOR "Use Boolector solver (default: ON)" ON)
option(ENABLE_Z3 "Use Z3 solver (default: ON)" ON)
option(ENABLE_MATHSAT "Use MathSAT solver (default: OFF)" OFF)
option(ENABLE_YICES "Use Yices solver (default: OFF)" OFF)

#############################
# OTHERS
#############################
option(ENABLE_LIBM "Use libm from c2goto (default: ON)" ON)
option(ENABLE_FUZZER "Add fuzzing targets (default: OFF)" OFF)
#############################
# CMake extra Vars
#############################
set(Clang_DIR "${LLVM_DIR}" CACHE STRING "Clang Directory (if not set, this will be set to the LLVM_DIR")  

# Demand C++14
set (CMAKE_CXX_STANDARD 14)

# Used by try_compile
set(CMAKE_POSITION_INDEPENDENT_CODE ON)