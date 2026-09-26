#################################
# Project Information           #
#################################
cmake_minimum_required(VERSION 3.24)

foreach(policy CMP0135 CMP0167 CMP0169)
  if(POLICY ${policy})
    cmake_policy(SET ${policy} NEW)
  endif()
endforeach()

# VERSION also sets ESBMC_VERSION and ESBMC_VERSION_{MAJOR,MINOR,PATCH}, which
# cmake_config.in and CPack read.
project(ESBMC
  VERSION 8.5.0
  DESCRIPTION "Efficient SMT-based Context-Bounded Model Checker"
  HOMEPAGE_URL "https://github.com/esbmc/esbmc"
  LANGUAGES C CXX)

# Solver subdirectories append to this during configuration.
set(ESBMC_AVAILABLE_SOLVERS "")

set(MIN_SUPPORTED_LLVM_VERSION_MAJOR 18)
set(MAX_SUPPORTED_LLVM_VERSION_MAJOR 22)

#################################
# Configuration                 #
#################################

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# TODO: adopt C++20 modules once cmake_minimum_required reaches 3.28 and ccache
# can cache module compilations (it cannot as of 4.14).
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Adds custom modules from ESBMC and default Options
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/scripts/cmake/")

# Set a default build type if none was specified
set(default_build_type "RelWithDebInfo")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE
      STRING "Choose the type of build." FORCE)
endif()

# Set the possible values of build type for cmake-gui
set(CMAKE_CXX_FLAGS_DEBUGOPT "-O2 -g" CACHE STRING "C++ compiler flags for DebugOpt build" FORCE)
set(CMAKE_C_FLAGS_DEBUGOPT "-O2 -g" CACHE STRING "C compiler flags for DebugOpt build" FORCE)
mark_as_advanced(
  CMAKE_CXX_FLAGS_DEBUGOPT
  CMAKE_C_FLAGS_DEBUGOPT
)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
             "Debug" "Release" "MinSizeRel" "RelWithDebInfo" "Sanitizer" "DebugOpt")

set(ESBMC_CXX_Clang_DEFAULTS -Wall -Wextra -pipe)
set(ESBMC_CXX_GNU_DEFAULTS -Wall -Wextra -pipe)
add_compile_options(${ESBMC_CXX_${CMAKE_CXX_COMPILER_ID}_DEFAULTS})

include(CheckIncludeFile)
check_include_file(unistd.h HAVE_UNISTD)

include(Utils)
include(OSConfiguration)
include(Options)
include(SendFileHack)
include(InstallFiles)
include(Sanitizers)
include(ClangTidy)
include(Coverage)

if(BUILD_STATIC)
  include(BuildStatic)
endif()

if(CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
    # Homebrew's static Boost.Iostreams injects bare -l names for its
    # compression/ICU backends (zstd + icu{data,i18n,uc}) with no search path,
    # so the static link fails with "library 'zstd' not found" (esbmc/esbmc#6234).
    # zstd is symlinked into $(brew --prefix)/lib; icu4c is keg-only. Add both as
    # link dirs — CMAKE_PREFIX_PATH cannot help, the names bypass find_library.
    if(BUILD_STATIC)
        foreach(brew_keg "" icu4c)
            execute_process(
                COMMAND brew --prefix ${brew_keg}
                RESULT_VARIABLE BREW_KEG_RESULT
                OUTPUT_VARIABLE BREW_KEG_PREFIX
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
            if(BREW_KEG_RESULT EQUAL 0 AND IS_DIRECTORY "${BREW_KEG_PREFIX}/lib")
                message(STATUS "Adding Homebrew link directory ${BREW_KEG_PREFIX}/lib")
                link_directories("${BREW_KEG_PREFIX}/lib")
            endif()
        endforeach()
    endif()
endif()

# This MUST be executed after BuildStatic since it sets Boost Static flags
# In boost-1.89, system and filesystem has been remove
# To support the older version, added "optional" here
find_package(Boost REQUIRED
  COMPONENTS date_time program_options iostreams
  OPTIONAL_COMPONENTS system filesystem)
include(SetupLocalLLVM)

include(ExternalDependencies)

include(SetupMimalloc)

include(SetupImmer)

# Optimization
include(OptimizationCCache)
include(BuildOptimizations)
include(WError)

# The include paths, definitions and Boost dependency every ESBMC target shares.
# Must follow find_package(Boost) and the dependency setup above.
include(EsbmcCommon)

add_subdirectory(src)

include(Irep2Optimization)

# Generate ac_config.h. This must be generated after solvers
configure_file(
 "${PROJECT_SOURCE_DIR}/scripts/cmake/cmake_config.in"
 "${PROJECT_BINARY_DIR}/src/ac_config.h"
)

# This should be added after all source files
include(Docs)

include(CTest) # calls enable_testing()
if(BUILD_TESTING)
    add_subdirectory(unit)
    if(ENABLE_LD_FRONTEND)
        add_subdirectory(regression/ld)
    endif()
endif()
# Added after enable_testing() so the ld-verify CLI's own CTest cases register;
# the binary itself is built whenever the LD front-end is enabled.
if(ENABLE_LD_FRONTEND)
  add_subdirectory(tools/ld-verify)
endif()
if(ENABLE_REGRESSION)
    add_subdirectory(regression)
endif()
include(SetupCsmith)
