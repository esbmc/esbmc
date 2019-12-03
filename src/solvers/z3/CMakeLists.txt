# Try and find z3. Search in priority of:
#  * User specified directories
#  * Environmental var paths
#  * System default
function(findz3 versionstr libz3 incz3)
    if (BUILD_STATIC)
        find_library(LIBZ3 NAMES libz3.a PATHS "${ARGN}" PATH_SUFFIXES lib bin NO_DEFAULT_PATH)
    else ()
        find_library(LIBZ3 NAMES libz3.so PATHS "${ARGN}" PATH_SUFFIXES lib bin NO_DEFAULT_PATH)
    endif ()
    find_path(INCZ3 z3.h PATHS "${ARGN}/include" NO_DEFAULT_PATHcd)

    if ((LIBZ3 STREQUAL "LIBZ3-NOTFOUND") OR (INCZ3 STREQUAL "INCZ3-NOTFOUND"))
        return()
    endif ()

    get_filename_component(Z3_LIBDIR "${LIBZ3}" DIRECTORY)
    try_run(Z3RUNS Z3COMPILES ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/try_z3.c
            CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${ARGN}/include
            LINK_LIBRARIES ${LIBZ3} ${LIBGMP} stdc++ m
            COMPILE_OUTPUT_VARIABLE Z3COMPILESTR
            RUN_OUTPUT_VARIABLE Z3VERSIONSTR)

    if ((NOT Z3RUNS EQ 0) OR (NOT Z3COMPILES))
        if (DEBUG_TRY_RUNS)
            message(FATAL_ERRROR "Failed Z3 compilation compile: ${Z3COMPILESTR}")
        endif ()
        return()
    endif ()

    set(${libz3} "${LIBZ3}" PARENT_SCOPE)
    set(${incz3} "${ARGN}/include" PARENT_SCOPE)
    set(${versionstr} "${Z3VERSIONSTR}" PARENT_SCOPE)
endfunction()

if (DEFINED DISABLE_Z3)
    if (DEFINED ENABLE_Z3)
        message(SEND_ERROR "Can't request both disable and enable z3")
    endif ()
    set(YESWEUSED_DIABLE_Z3, "${DISABLE_Z3}")
    return()
endif ()

set(LIBZ3 "LIBZ3-NOTFOUND")
set(INCZ3 "LIBZ3-NOTFOUND")

if (DEFINED Z3_DIR)
    findz3(Z3VERSIONSTR LIBZ3 INCZ3 "${Z3_DIR}")
    if (LIBZ3 STREQUAL "LIBZ3-NOTFOUND")
        message(SEND_ERROR "Couldn't find Z3 where you pointed Z3_DIR: ${Z3_DIR}")
    endif ()
endif ()

if ((LIBZ3 STREQUAL "LIBZ3-NOTFOUND") AND (NOT $ENV{SATDIR64} STREQUAL ""))
    findz3(Z3VERSIONSTR LIBZ3 INCZ3 "$ENV{SATDIR64}/z3")
endif ()

if ((LIBZ3 STREQUAL "LIBZ3-NOTFOUND") AND (NOT $ENV{SATDIR32} STREQUAL ""))
    findz3(Z3VERSIONSTR LIBZ3 INCZ3 "$ENV{SATDIR32}/z3")
endif ()

if ((LIBZ3 STREQUAL LIBZ3 INCZ3 "LIBZ3-NOTFOUND") AND (NOT $ENV{SATDIR} STREQUAL ""))
    findz3(Z3VERSIONSTR LIBZ3 INCZ3 "$ENV{SATDIR}/z3")
endif ()

if (LIBZ3 STREQUAL "LIBZ3-NOTFOUND")
    findz3(Z3VERSIONSTR LIBZ3 INCZ3 "$ENV{HOME}/z3")
endif ()

if (LIBZ3 STREQUAL "LIBZ3-NOTFOUND")
    # Try default search instead
    find_library(LIBZ3 z3)
    set(Z3VERSIONSTR "(system)")
endif ()

if (LIBZ3 STREQUAL "LIBZ3-NOTFOUND")
    if (DEFINED ENABLE_Z3)
        message(SEND_ERROR "Z3 requested, but couldn't find it; please point Z3_DIR at it")
    endif ()
    # Otherwise, Z3 neither found nor requested, so ignore it.
else ()
    # Got it; build stuff.
    add_library(solverz3 z3_conv.cpp)
    target_include_directories(solverz3
            PRIVATE ${INCZ3}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    set(ENABLE_Z3 "1")
    message(STATUS "Z3 version: ${Z3VERSIONSTR}")
    if (BUILD_STATIC)
        target_link_libraries(solverz3 "${LIBZ3}" -pthread -lgomp -ldl)
    else ()
        target_link_libraries(solverz3 "${LIBZ3}")
    endif ()
    # Add to solver link
    target_link_libraries(solvers INTERFACE solverz3)

    set(ESBMC_ENABLE_z3 1 PARENT_SCOPE)
endif ()