#################################
# Project Information           #
#################################
cmake_minimum_required (VERSION 3.7)
project (ESBMC)

set (ESBMC_VERSION_MAJOR 6)
set (ESBMC_VERSION_MINOR 0)
set (ESBMC_VERSION_PATCH 0)
set (ESBMC_VERSION_HEX 060000)
set (ESBMC_VERSION "${ESBMC_VERSION_MAJOR}.${ESBMC_VERSION_MINOR}.${ESBMC_VERSION_PATCH}")

# The only default solver available is smtlib
set (ESBMC_AVAILABLE_SOLVERS "smtlib")

add_subdirectory(src)

if(ENABLE_REGRESSION)
    add_subdirectory(regression)
endif()