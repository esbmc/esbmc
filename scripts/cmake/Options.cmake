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
option(BUILD_STATIC "Build ESBMC in static mode (default: OFF)" OFF)
option(BUILD_DOC "Build ESBMC documentation" OFF)
option(ENABLE_REGRESSION "Add Regressions Tests (default: OFF)" OFF)
option(ENABLE_COVERAGE "Generate Coverage Report (default: OFF)" OFF)
option(ENABLE_OLD_FRONTEND "Enable flex/bison language frontend (default: OFF)" OFF)
option(ENABLE_SOLIDITY_FRONTEND "Enable Solidity language frontend (default: OFF)" OFF)
option(ENABLE_GOTO_CONTRACTOR "Enable IBEX in the build (default: OFF)" OFF)
option(ENABLE_JIMPLE_FRONTEND "Enable Jimple language frontend (default: OFF)" OFF)

#############################
# SOLVERS
#############################
option(ENABLE_BOOLECTOR "Use Boolector solver (default: OFF)" OFF)
option(ENABLE_Z3 "Use Z3 solver (default: OFF)" OFF)
option(ENABLE_MATHSAT "Use MathSAT solver (default: OFF)" OFF)
option(ENABLE_YICES "Use Yices solver (default: OFF)" OFF)
option(ENABLE_CVC4 "Use CVC4 solver (default: OFF)" OFF)
option(ENABLE_BITWUZLA "Use Bitwuzla solver (default: OFF)" OFF)

#############################
# OTHERS
#############################
option(ESBMC_BUNDLE_LIBC "Use libc from c2goto (default: ON)" ON)
option(ENABLE_LIBM "Use libm from c2goto (default: ON)" ON)
option(ENABLE_FUZZER "Add fuzzing targets (default: OFF)" OFF)
option(ENABLE_CLANG_TIDY "Activate clang tidy analysis (default: OFF)" OFF)
option(ENABLE_CSMITH "Add csmith Tests (default: OFF) (depends: ENABLE_REGRESSION)" OFF)
option(BENCHBRINGUP "Run a user-specified benchmark in Github workflow" OFF)
option(DOWNLOAD_DEPENDENCIES "Download and build dpendencies if needed (default: OFF)" OFF)

#############################
# PRE-BUILT DEPENDENCIES
#############################
if(WIN32)
set(DEFAULT_LLVM_URL "https://gitlab.com/Anthonysdu/llvm11/-/raw/main/llvm+clang+lld-11.0.0-x86_64-windows-msvc-release-mt.zip")
set(DEFAULT_LLVM_NAME "llvm+clang+lld-11.0.0-x86_64-windows-msvc-release-mt")

set(DEFAULT_Z3_URL "https://github.com/Z3Prover/z3/releases/download/z3-4.12.2/z3-4.12.2-x86-win.zip")
set(DEFAULT_Z3_NAME z3-4.12.2-x86-win)

else()
set(DEFAULT_LLVM_URL "https://github.com/llvm/llvm-project/releases/download/llvmorg-11.0.0/clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04.tar.xz")
set(DEFAULT_LLVM_NAME "clang+llvm-11.0.0-x86_64-linux-gnu-ubuntu-20.04")

set(DEFAULT_Z3_URL "https://github.com/Z3Prover/z3/releases/download/z3-4.12.2/z3-4.12.2-x64-glibc-2.35.zip")
set(DEFAULT_Z3_NAME z3-4.12.2-x64-glibc-2.35)
endif()

set(ESBMC_LLVM_URL ${DEFAULT_LLVM_URL} CACHE STRING "URL to download prebuilt LLVM")
set(ESBMC_LLVM_NAME ${DEFAULT_LLVM_NAME} CACHE STRING "Name of the extracted directory of LLVM")

set(ESBMC_Z3_URL ${DEFAULT_LLVM_URL} CACHE STRING "URL to download prebuilt Z3")
set(ESBMC_Z3_NAME ${DEFAULT_LLVM_NAME} CACHE STRING "Name of the extracted directory of Z3")

#############################
# CMake extra Vars
#############################
# ESBMC_CLANG_HEADERS_BUNDLED: 'detect', On, Off
set(ESBMC_CLANG_HEADERS_BUNDLED "detect" CACHE STRING "Bundle the Clang resource-dir headers (default: detect)")
set(OVERRIDE_CLANG_HEADER_DIR "")
set(Clang_DIR "${LLVM_DIR}" CACHE STRING "Clang Directory (if not set, this will be set to the LLVM_DIR")

# Demand C++17
set (CMAKE_CXX_STANDARD 17)

# Used by try_compile
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
