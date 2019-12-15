# Mathsat depends on gmp
find_library(LIBGMP gmp)
if ((LIBGMP STREQUAL "LIBGMP-NOTFOUND") AND (DEFINED MSAT_DIR))
    find_library(LIBGMP gmp PATHS "${MSAT_DIR}")
endif ()

if (LIBGMP STREQUAL "LIBGMP-NOTFOUND")
    if (DEFINED ENABLE_MSAT)
        message(FATAL_ERROR "Requested mathsat, however I can't find libgmp in system or \\$MSAT_DIR")
    else ()
        message(WARNING "Skipping searching for mathsat as I can't find libgmp, try pointing \\$MSAT_DIR at its director")
        return()
    endif ()
endif ()


function(findmsat versionstr libmsat incmsat)
    find_library(LIBMSAT mathsat NAMES libmathsat PATHS "${ARGN}/lib" NO_DEFAULT_PATH)
    find_path(INCMSAT mathsat.h PATHS "${ARGN}/include" NO_DEFAULT_PATH)
    if ((LIBMSAT STREQUAL "LIBMSAT-NOTFOUND") OR (INCMSAT STREQUAL "INCMSAT-NOTFOUND"))
        return()
    endif ()

    get_filename_component(MSAT_LIBDIR "${LIBMSAT}" DIRECTORY)
    try_run(MSATRUNS MSATCOMPILES ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/try_msat.c
            CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${ARGN}/include -DLINK_DIRECTORIES=${MSAT_LIBDIR}
            LINK_LIBRARIES ${LIBMSAT} ${LIBGMP} stdc++ m
            COMPILE_OUTPUT_VARIABLE MSATCOMPILESTR
            RUN_OUTPUT_VARIABLE MSATVERSIONSTR)

    if ((NOT MSATRUNS EQ 0) OR (NOT MSATCOMPILES))
        if (DEBUG_TRY_RUNS)
            message("Failed mathsat compilation compile: ${MSATCOMPILESTR}")
        endif ()
        return()
    endif ()

    set(${libmsat} "${LIBMSAT}" PARENT_SCOPE)
    set(${incmsat} "${ARGN}/include" PARENT_SCOPE)
    set(${versionstr} "${MSATVERSIONSTR}" PARENT_SCOPE)
endfunction()

if (DEFINED DISABLE_MSAT)
    if (DEFINED ENABLE_MSAT)
        message(SEND_ERROR "Can't request both disable and enable msat")
    endif ()
    set(YESWEUSED_DIABLE_MSAT, "${DISABLE_MSAT}")
    return()
endif ()


set(LIBMSAT "LIBMSAT-NOTFOUND")
set(INCMSAT "LIBMSAT-NOTFOUND")

if (DEFINED MSAT_DIR)
    findmsat(MSATVERSIONSTR LIBMSAT INCMSAT "${MSAT_DIR}")
    if (LIBMSAT STREQUAL "LIBMSAT-NOTFOUND")
        message(SEND_ERROR "Couldn't find mathsat where you pointed MSAT_DIR: ${MSAT_DIR}")
    endif ()
endif ()

if ((LIBMSAT STREQUAL "LIBMSAT-NOTFOUND") AND (NOT $ENV{SATDIR64} STREQUAL ""))
    findmsat(MSATVERSIONSTR LIBMSAT INCMSAT "$ENV{SATDIR64}/mathsat")
endif ()

if ((LIBMSAT STREQUAL "LIBMSAT-NOTFOUND") AND (NOT $ENV{SATDIR32} STREQUAL ""))
    findmsat(MSATVERSIONSTR LIBMSAT INCMSAT "$ENV{SATDIR32}/mathsat")
endif ()

if ((LIBMSAT STREQUAL LIBMSAT INCMSAT "LIBMSAT-NOTFOUND") AND (NOT $ENV{SATDIR} STREQUAL ""))
    findmsat(MSATVERSIONSTR LIBMSAT INCMSAT "$ENV{SATDIR}/mathsat")
endif ()

if (LIBMSAT STREQUAL "LIBMSAT-NOTFOUND")
    findmsat(MSATVERSIONSTR LIBMSAT INCMSAT "$ENV{HOME}/mathsat")
endif ()

if (LIBMSAT STREQUAL "LIBMSAT-NOTFOUND")
    # Try default search instead
    find_library(LIBMSAT mathsat)
    set(MSATVERSIONSTR "(system)")
endif ()

if (LIBMSAT STREQUAL "LIBMSAT-NOTFOUND")
    if (DEFINED ENABLE_MSAT)
        message(SEND_ERROR "MSAT requested, but couldn't find it; please point MSAT_DIR at it")
    endif ()
    # Otherwise, MSAT neither found nor requested, so ignore it.
else ()
    # Got it; build stuff.
    add_library(solvermsat mathsat_conv.cpp)
    target_include_directories(solvermsat
            PRIVATE ${INCMSAT}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(solvermsat "${LIBMSAT}" ${LIBGMP})
    set(ENABLE_MSAT "1")
    message(STATUS "MathSAT version: ${MSATVERSIONSTR}")

    # Add to solver link
    target_link_libraries(solvers INTERFACE solvermsat)

    set(ESBMC_ENABLE_mathsat 1 PARENT_SCOPE)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} mathsat" PARENT_SCOPE)
endif ()