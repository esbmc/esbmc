set(Mathsat_MIN_VERSION 5.5.4)

if (DEFINED Mathsat_DIR)
    set(ENABLE_MATHSAT ON PARENT_SCOPE)
elseif (EXISTS $ENV{HOME}/mathsat)
  set(ENABLE_MATHSAT ON PARENT_SCOPE)
else()
  if(ENABLE_MATHSAT AND DOWNLOAD_DEPENDENCIES AND ACADEMIC_BUILD)
    message("Downloading Mathsat")
    download_zip_and_extract(MATHSAT ${MATHSAT_URL})
    set(Mathsat_DIR ${CMAKE_BINARY_DIR}/MATHSAT/${MATHSAT_NAME})
  endif()
endif ()

if (ENABLE_MATHSAT)
    find_library(Mathsat_LIB mathsat HINTS "${Mathsat_DIR}" $ENV{HOME}/mathsat PATH_SUFFIXES lib)
    find_path(Mathsat_INCLUDE_DIRS mathsat.h HINTS ${Mathsat_DIR} $ENV{HOME}/mathsat PATH_SUFFIXES include)

    if (Mathsat_INCLUDE_DIRS STREQUAL "Mathsat_INCLUDE_DIRS-NOTFOUND")
        message(FATAL_ERROR "Could not find mathsat include headers, please check Mathsat_DIR")
    endif ()

    if (Mathsat_LIB STREQUAL "Mathsat_LIB-NOTFOUND")
        message(FATAL_ERROR "Could not find libmathsat, please check Mathsat_DIR")
    endif ()

    find_library(GMP_LIB gmp)
    try_run(MSATRUNS Mathsat_COMPILES ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/try_msat.c
            CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${Mathsat_INCLUDE_DIRS}
            LINK_LIBRARIES ${Mathsat_LIB} ${GMP_LIB} stdc++ m
            COMPILE_OUTPUT_VARIABLE MSATCOMPILESTR
            RUN_OUTPUT_VARIABLE Mathsat_VERSION)

    if (NOT Mathsat_COMPILES)
        message(FATAL_ERROR "Could not use MathSAT: \n${MSATCOMPILESTR}")
    endif ()

    message(STATUS "Using MathSAT at: ${Mathsat_LIB}")
    string(REGEX MATCH "version ([0-9]+\.[0-9]+\.[0-9]+) " REGEX_OUTPUT ${Mathsat_VERSION})
    set(Mathsat_VERSION "${CMAKE_MATCH_1}")
    message(STATUS "MathSAT version: ${Mathsat_VERSION}")
    if (Mathsat_VERSION VERSION_LESS Mathsat_MIN_VERSION)
        message(FATAL_ERROR "Expected version ${Mathsat_MIN_VERSION} or greater")
    endif ()

    add_library(solvermsat mathsat_conv.cpp)
    target_include_directories(solvermsat
            PRIVATE ${Mathsat_INCLUDE_DIRS}
            PRIVATE ${Boost_INCLUDE_DIRS}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(solvermsat "${Mathsat_LIB}" fmt::fmt)

    # Add to solver link
    target_link_libraries(solvers INTERFACE solvermsat)

    set(ESBMC_ENABLE_mathsat 1 PARENT_SCOPE)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} mathsat" PARENT_SCOPE)
endif ()
