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
option(ENABLE_SOLIDITY_FRONTEND "Enable Solidity language frontend (default: OFF)" OFF)
option(ENABLE_GOTO_CONTRACTOR "Enable IBEX in the build (default: OFF)" OFF)
option(ENABLE_JIMPLE_FRONTEND "Enable Jimple language frontend (default: OFF)" OFF)
option(ENABLE_PYTHON_FRONTEND "Enable Python language frontend (default: OFF)" OFF)

#############################
# SOLVERS
#############################
option(ENABLE_SMTLIB "Use SMTLIB interface solver (default: ON)" ON)
option(ENABLE_BOOLECTOR "Use Boolector solver (default: OFF)" OFF)
option(ENABLE_Z3 "Use Z3 solver (default: OFF)" OFF)
option(ENABLE_MATHSAT "Use MathSAT solver (default: OFF)" OFF)
option(ENABLE_YICES "Use Yices solver (default: OFF)" OFF)
option(ENABLE_CVC4 "Use CVC4 solver (default: OFF)" OFF)
option(ENABLE_CVC5 "Use CVC5 solver (default: OFF)" OFF)
option(ENABLE_BITWUZLA "Use Bitwuzla solver (default: OFF)" OFF)
option(ENABLE_BITWUZLLOB "Use Bitwuzllob (Bitwuzla on the Mallob platform) via an external mallob binary (default: ON)" ON)
option(ENABLE_NEUROSYM "Use NeuroSym (neural-guided GAN + Z3 fallback) via an external Python program (default: ON)" ON)

#############################
# OTHERS
#############################
option(ESBMC_BUNDLE_LIBC "Use libc from c2goto (default: ON)" ON)
option(ENABLE_BUNDLE_LIBC_32BIT "Use 32-bits libc from c2goto (default: ON)" ON)
option(ENABLE_LIBM "Use libm from c2goto (default: ON)" ON)
option(ENABLE_FUZZER "Add fuzzing targets (default: OFF)" OFF)
option(ENABLE_CLANG_TIDY "Activate clang tidy analysis (default: OFF)" OFF)
option(ENABLE_CSMITH "Add csmith Tests (default: OFF) (depends: ENABLE_REGRESSION)" OFF)
option(BENCHBRINGUP "Run a user-specified benchmark in Github workflow" OFF)
option(DOWNLOAD_DEPENDENCIES "Download and build dependencies if needed (default: OFF)" OFF)
option(ENABLE_MIMALLOC "Link the mimalloc allocator into esbmc (default: OFF). Speeds up the allocation-heavy symex path (~15% on high-unwind runs) but regresses some SV-COMP benchmarks. Found via find_package, or downloaded when DOWNLOAD_DEPENDENCIES is ON." OFF)
option(ACADEMIC_BUILD "Check and Enable libs that available only in Academic builds (default: OFF)" OFF)
option(CORE_REGRESSION_ONLY "Only add tests in the regression that are CORE (default: OFF)" OFF)

#############################
# PRE-BUILT DEPENDENCIES
#############################

# Pre-built LLVM toolchains. Windows ships upstream's x86_64 archive; Linux
# x86_64 uses a trimmed archive on the ESBMC release, Linux aarch64 uses the
# trimmed archive produced by esbmc/llvm (same LLVM major, 22).
if(WIN32)
  set(DEFAULT_LLVM_URL "https://github.com/llvm/llvm-project/releases/download/llvmorg-22.1.4/clang+llvm-22.1.4-x86_64-pc-windows-msvc.tar.xz")
  set(DEFAULT_LLVM_NAME "clang+llvm-22.1.4-x86_64-pc-windows-msvc")

  set(DEFAULT_Z3_URL "https://github.com/Z3Prover/z3/releases/download/z3-4.13.3/z3-4.13.3-x64-win.zip")
  set(DEFAULT_Z3_NAME z3-4.13.3-x64-win)

  set(MATHSAT_URL "https://mathsat.fbk.eu/download.php?file=mathsat-5.6.10-win64-msvc.zip")
  set(MATHSAT_NAME "mathsat-5.6.10-win64-msvc")
elseif(NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "^(aarch64|arm64)$")
  set(ESBMC_AARCH64_LINUX TRUE)

  # Trimmed from upstream LLVM-22.1.6-Linux-ARM64 by esbmc/llvm's tools/trim.sh
  # (see the Build trimmed LLVM archive workflow there). The archive's top-level
  # directory is preserved verbatim so SetupLocalLLVM's
  # ${CMAKE_BINARY_DIR}/LLVM/${ESBMC_LLVM_NAME} extraction contract holds. Z3
  # comes from its arm64 prebuilt, which ships the libz3.a a static link needs.
  set(DEFAULT_LLVM_URL "https://github.com/esbmc/llvm/releases/download/llvm-22.1.6-armv8/LLVM-22.1.6-Linux-ARM64-esbmc.tar.xz")
  set(DEFAULT_LLVM_NAME "LLVM-22.1.6-Linux-ARM64")
  set(DEFAULT_Z3_URL "https://github.com/Z3Prover/z3/releases/download/z3-4.13.3/z3-4.13.3-arm64-glibc-2.34.zip")
  set(DEFAULT_Z3_NAME z3-4.13.3-arm64-glibc-2.34)
else()
  set(DEFAULT_LLVM_URL "https://github.com/esbmc/esbmc/releases/download/v8.3/clang+llvm-22.1.6-x86_64-linux-gnu-ubuntu-22.04.tar.xz")
  set(DEFAULT_LLVM_NAME "clang+llvm-22.1.6-x86_64-linux-gnu-ubuntu-22.04")

  set(DEFAULT_CHERI_LLVM_URL "https://github.com/XLiZHI/esbmc/releases/download/v17/clang-cheri-17.zip")
  set(DEFAULT_CHERI_LLVM_NAME "clang-cheri-17")

  set(DEFAULT_Z3_URL "https://github.com/Z3Prover/z3/releases/download/z3-4.13.3/z3-4.13.3-x64-glibc-2.35.zip")
  set(DEFAULT_Z3_NAME z3-4.13.3-x64-glibc-2.35)

  set(MATHSAT_URL "https://mathsat.fbk.eu/release/mathsat-5.6.11-linux-x86_64.tar.gz")
  set(MATHSAT_NAME "mathsat-5.6.11-linux-x86_64")

  set(DEFAULT_CVC5_URL "https://github.com/cvc5/cvc5/releases/download/cvc5-1.1.2/cvc5-Linux-static.zip")
  set(DEFAULT_CVC5_NAME cvc5-Linux-static)
endif()

set(ESBMC_LLVM_URL ${DEFAULT_LLVM_URL} CACHE STRING "URL to download prebuilt LLVM")
set(ESBMC_LLVM_NAME ${DEFAULT_LLVM_NAME} CACHE STRING "Name of the extracted directory of LLVM")

set(ESBMC_Z3_URL ${DEFAULT_Z3_URL} CACHE STRING "URL to download prebuilt Z3")
set(ESBMC_Z3_NAME ${DEFAULT_Z3_NAME} CACHE STRING "Name of the extracted directory of Z3")

set(ESBMC_CVC5_URL ${DEFAULT_CVC5_URL} CACHE STRING "URL to download prebuilt CVC")
set(ESBMC_CVC5_NAME ${DEFAULT_CVC5_NAME} CACHE STRING "Name of the extracted directory of CVC")

set(ESBMC_CHERI_LLVM_URL ${DEFAULT_CHERI_LLVM_URL} CACHE STRING "URL to download prebuilt CHERI LLVM")
set(ESBMC_CHERI_LLVM_NAME ${DEFAULT_CHERI_LLVM_NAME} CACHE STRING "Name of the extracted directory of CHERI LLVM")

#############################
# CMake extra Vars
#############################
# ESBMC_CLANG_HEADERS_BUNDLED: 'detect', On, Off
set(ESBMC_CLANG_HEADERS_BUNDLED "detect" CACHE STRING "Bundle the Clang resource-dir headers (default: detect)")
set(OVERRIDE_CLANG_HEADER_DIR "")
set(Clang_DIR "${LLVM_DIR}" CACHE STRING "Clang Directory (if not set, this will be set to the LLVM_DIR")

set(ESBMC_CHERI "" CACHE STRING "override CHERI-support enabled through detection of Clang")
set(ESBMC_CHERI_HYBRID_SYSROOT "" CACHE STRING "Path containing the mips64-unknown-freebsd sysroot typically generated by 'cheribuild.py cheribsd-mips64-hybrid'")
set(ESBMC_CHERI_PURECAP_SYSROOT "" CACHE STRING "Path containing the mips64-unknown-freebsd sysroot typically generated by 'cheribuild.py cheribsd-mips64-purecap'")

# CHERI, CVC5 and MathSAT publish no aarch64 build, and an unset URL would
# otherwise reach download_zip_and_extract as an empty string. Say what is
# unsupported instead. See https://github.com/esbmc/esbmc/issues/7230.
if(ESBMC_AARCH64_LINUX)
  if(ESBMC_CHERI)
    message(FATAL_ERROR "CHERI is not supported on aarch64: no CHERI-enabled Clang is published for this architecture.")
  endif()
  if(ENABLE_CVC5 AND DOWNLOAD_DEPENDENCIES)
    message(FATAL_ERROR "CVC5 is not supported on aarch64: cvc5 1.1.2 publishes no Linux arm64 build. Pass -DENABLE_CVC5=Off.")
  endif()
  if(ENABLE_MATHSAT AND DOWNLOAD_DEPENDENCIES)
    message(FATAL_ERROR "MathSAT is not supported on aarch64: no Linux arm64 build is published. Pass -DENABLE_MATHSAT=Off.")
  endif()
endif()

set(ESBMC_BUNDLE_LIBC_32BIT "${ENABLE_BUNDLE_LIBC_32BIT}" CACHE BOOL "Enable 32-bit libc bundling" FORCE)

# Demand C++23
set(CMAKE_CXX_STANDARD 23)

# Used by try_compile
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
