if(DEFINED Mathsat_DIR)
    set(ENABLE_MATHSAT ON)
endif()

if(EXISTS $ENV{HOME}/mathsat)
    set(ENABLE_MATHSAT ON)
endif()

if(ENABLE_MATHSAT)
    find_library(Mathsat_LIB mathsat PATHS "${Mathsat_DIR}" $ENV{HOME}/mathsat PATH_SUFFIXES lib)
    find_path(Mathsat_INCLUDE_DIRS mathsat.h PATHS ${Mathsat_DIR} $ENV{HOME}/mathsat PATH_SUFFIXES include)

    if(Mathsat_INCLUDE_DIRS STREQUAL "Mathsat_INCLUDE_DIRS-NOTFOUND")
        message(FATAL_ERROR "Could not find mathsat include headers, please check Mathsat_DIR")
    endif()

    if(Mathsat_LIB STREQUAL "Mathsat_LIB-NOTFOUND")
        message(FATAL_ERROR "Could not find libmathsat, please check Mathsat_DIR")
    endif()

    find_library(GMP_LIB gmp)
    try_run(MSATRUNS Mathsat_COMPILES ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/try_msat.c
            CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${Mathsat_INCLUDE_DIRS}
            LINK_LIBRARIES ${Mathsat_LIB} ${GMP_LIB} stdc++ m
            COMPILE_OUTPUT_VARIABLE MSATCOMPILESTR
            RUN_OUTPUT_VARIABLE Mathsat_VERSION)

    if(NOT Mathsat_COMPILES)
        message(FATAL_ERROR "Could not use MathSAT: \n${MSATCOMPILESTR}")
    endif()

    message(STATUS "Using MathSAT at: ${Mathsat_LIB}")
    string(REGEX MATCH "version ([0-9].[0-9].[0-9]) "  REGEX_OUTPUT ${Mathsat_VERSION})
    message(STATUS "MathSAT version: ${CMAKE_MATCH_1}")

    add_library(solvermsat mathsat_conv.cpp)
    target_include_directories(solvermsat
            PRIVATE ${Mathsat_INCLUDE_DIRS}
            PRIVATE ${Boost_INCLUDE_DIRS}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(solvermsat "${Mathsat_LIB}" ${GMP_LIB})

    # Add to solver link
    target_link_libraries(solvers INTERFACE solvermsat)

    set(ESBMC_ENABLE_mathsat 1 PARENT_SCOPE)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} mathsat" PARENT_SCOPE)
endif()