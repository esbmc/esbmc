function(findyices versionstr libyices incyices)
    find_library(LIBYICES yices NAMES libyices.so PATHS "${ARGN}" PATH_SUFFIXES lib NO_DEFAULT_PATH)
    find_path(INCYICES yices.h PATHS "${ARGN}/include" NO_DEFAULT_PATH)
    if ((LIBYICES STREQUAL "LIBYICES-NOTFOUND") OR (INCYICES STREQUAL "INCYICES-NOTFOUND"))
        return()
    endif ()

    get_filename_component(YICES_LIBDIR "${LIBYICES}" DIRECTORY)
    try_run(YICESRUNS YICESCOMPILES ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/try_yices.c
            CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${ARGN}/include -DLINK_DIRECTORIES=${YICES_LIBDIR}
            LINK_LIBRARIES yices
            COMPILE_OUTPUT_VARIABLE YICESCOMPILESTR
            RUN_OUTPUT_VARIABLE YICESVERSIONSTR)

    if ((NOT YICESRUNS EQ 0) OR (NOT YICESCOMPILES))
        if (DEBUG_TRY_RUNS)
            message("Failed yices compilation compile: ${YICESCOMPILESTR}")
        endif ()
        return()
    endif ()

    set(${libyices} "${LIBYICES}" PARENT_SCOPE)
    set(${incyices} "${ARGN}/include" PARENT_SCOPE)
    set(${versionstr} "${YICESVERSIONSTR}" PARENT_SCOPE)
endfunction()

if (DEFINED DISABLE_YICES)
    if (DEFINED ENABLE_YICES)
        message(SEND_ERROR "Can't request both disable and enable yices")
    endif ()
    set(YESWEUSED_DIABLE_YICES, "${DISABLE_YICES}")
    return()
endif ()

set(LIBYICES "LIBYICES-NOTFOUND")
set(INCYICES "LIBYICES-NOTFOUND")

if (DEFINED YICES_DIR)
    findyices(YICESVERSIONSTR LIBYICES INCYICES "${YICES_DIR}")
    if (LIBYICES STREQUAL "LIBYICES-NOTFOUND")
        message(SEND_ERROR "Couldn't find YICES where you pointed YICES_DIR: ${YICES_DIR}")
    endif ()
endif ()

if ((LIBYICES STREQUAL "LIBYICES-NOTFOUND") AND (NOT $ENV{SATDIR64} STREQUAL ""))
    findyices(YICESVERSIONSTR LIBYICES INCYICES "$ENV{SATDIR64}/yices")
endif ()

if ((LIBYICES STREQUAL "LIBYICES-NOTFOUND") AND (NOT $ENV{SATDIR32} STREQUAL ""))
    findyices(YICESVERSIONSTR LIBYICES INCYICES "$ENV{SATDIR32}/yices")
endif ()

if ((LIBYICES STREQUAL LIBYICES INCYICES "LIBYICES-NOTFOUND") AND (NOT $ENV{SATDIR} STREQUAL ""))
    findyices(YICESVERSIONSTR LIBYICES INCYICES "$ENV{SATDIR}/yices")
endif ()

if (LIBYICES STREQUAL "LIBYICES-NOTFOUND")
    findyices(YICESVERSIONSTR LIBYICES INCYICES "$ENV{HOME}/yices")
endif ()

if (LIBYICES STREQUAL "LIBYICES-NOTFOUND")
    # Try default search instead
    find_library(LIBYICES yices)
    set(YICESVERSIONSTR "(system)")
endif ()

if (LIBYICES STREQUAL "LIBYICES-NOTFOUND")
    if (DEFINED ENABLE_YICES)
        message(SEND_ERROR "YICES requested, but couldn't find it; please point YICES_DIR at it")
    endif ()
    # Otherwise, YICES neither found nor requested, so ignore it.
else ()
    # Got it; build stuff.
    add_library(solveryices yices_conv.cpp)
    target_include_directories(solveryices
            PRIVATE ${INCYICES}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(solveryices "${LIBYICES}" gmp)
    set(ENABLE_YICES "1")
    message(STATUS "YICES version: ${YICESVERSIONSTR}")

    # Add to solver link
    target_link_libraries(solvers INTERFACE solveryices)

    if (YICESVERSIONSTR VERSION_GREATER 2.5.2)
        message(WARNING "Danger Will Robinson: the version of Yices you have is GPL licensed, distribution is impossible")
    endif ()

    set(ESBMC_ENABLE_yices 1 PARENT_SCOPE)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} yices" PARENT_SCOPE)
endif ()