# Module to find IBEX

# TODO: Check version
message(AUTHOR_WARNING "TODO: Ibex library relies on pkgconfig, we should parse it properly in future")

find_library(IBEX_LIB ibex libibex HINTS ${IBEX_DIR} $ENV{HOME}/ibex PATH_SUFFIXES lib bin)
find_library(IBEX_LIB_GAOL gaol libgaol HINTS ${IBEX_DIR} $ENV{HOME}/ibex PATH_SUFFIXES lib/ibex/3rd bin)
find_library(IBEX_LIB_GDTOA gdtoa libgdtoa HINTS ${IBEX_DIR} $ENV{HOME}/ibex PATH_SUFFIXES lib/ibex/3rd bin)
find_library(IBEX_LIB_ULTIM ultim libultim HINTS ${IBEX_DIR} $ENV{HOME}/ibex PATH_SUFFIXES lib/ibex/3rd bin)
find_library(IBEX_LIB_SOPLEX soplex libsoplex HINTS ${IBEX_DIR} $ENV{HOME}/ibex PATH_SUFFIXES lib/ibex/3rd bin)
find_path(IBEX_INCLUDE_DIRS ibex.h HINTS ${IBEX_DIR} $ENV{HOME}/ibex PATH_SUFFIXES include)

if(IBEX_INCLUDE_DIRS STREQUAL "IBEX_INCLUDE_DIRS-NOTFOUND")
    message(FATAL_ERROR "Could not find ibex include headers, please check IBEX_DIR")
endif()

if(IBEX_LIB STREQUAL "IBEX_LIB-NOTFOUND")
   message(FATAL_ERROR "Could not find libibex, please check IBEX_DIR")
endif()

if(IBEX_LIB_GAOL STREQUAL "IBEX_LIB_GAOL-NOTFOUND")
   message(FATAL_ERROR "Could not find libgaol, please check IBEX_DIR")
endif()

if(IBEX_LIB_GDTOA STREQUAL "IBEX_LIB_GDTOA-NOTFOUND")
   message(FATAL_ERROR "Could not find libgdtoa, please check IBEX_DIR")
endif()

if(IBEX_LIB_ULTIM STREQUAL "IBEX_LIB_ULTIM-NOTFOUND")
   message(FATAL_ERROR "Could not find libgultim, please check IBEX_DIR")
endif()

if(IBEX_LIB_SOPLEX STREQUAL "IBEX_LIB_SOPLEX-NOTFOUND")
   message(FATAL_ERROR "Could not find libsoplex, please check IBEX_DIR")
endif()

message(STATUS "Found Ibex at: ${IBEX_DIR}")