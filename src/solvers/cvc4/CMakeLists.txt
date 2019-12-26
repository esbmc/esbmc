function(findcvc4 versionstr libcvc4 inccvc4)
    find_library(LIBCVC4 cvc4 NAMES libcvc4 PATHS "${ARGN}" PATH_SUFFIXES lib NO_DEFAULT_PATH)
    find_path(INCCVC4 cvc4/cvc4.h PATHS "${ARGN}/include" NO_DEFAULT_PATH)
    if ((LIBCVC4 STREQUAL "LIBCVC4-NOTFOUND") OR (INCCVC4 STREQUAL "INCCVC4-NOTFOUND"))
        return()
    endif ()

    get_filename_component(CVC4_LIBDIR "${LIBCVC4}" DIRECTORY)

    try_run(CVC4RUNS CVC4COMPILES ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/try_cvc4.cpp
            CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${ARGN}/include -DLINK_DIRECTORIES=${CVC4_LIBDIR}
            LINK_LIBRARIES ${LIBCVC4} gmp
            COMPILE_OUTPUT_VARIABLE CVC4COMPILESTR
            RUN_OUTPUT_VARIABLE CVC4VERSIONSTR)

    if ((NOT CVC4RUNS EQ 0) OR (NOT CVC4COMPILES))
        if (DEBUG_TRY_RUNS)
            message("Failed cvc4 compilation compile: ${CVC4COMPILESTR}")
        endif ()

        return()
    endif ()

    set(${libcvc4} "${LIBCVC4}" PARENT_SCOPE)
    set(${inccvc4} "${ARGN}/include" PARENT_SCOPE)
    set(${versionstr} "${CVC4VERSIONSTR}" PARENT_SCOPE)
endfunction()

if (DEFINED DISABLE_CVC4)
    if (DEFINED ENABLE_CVC4)
        message(SEND_ERROR "Can't request both disable and enable cvc4")
    endif ()
    set(YESWEUSED_DIABLE_CVC4, "${DISABLE_CVC4}")
    return()
endif ()

set(LIBCVC4 "LIBCVC4-NOTFOUND")
set(INCCVC4 "LIBCVC4-NOTFOUND")

if (DEFINED CVC4_DIR)
    findcvc4(CVC4VERSIONSTR LIBCVC4 INCCVC4 "${CVC4_DIR}")
    if (LIBCVC4 STREQUAL "LIBCVC4-NOTFOUND")
        message(SEND_ERROR "Couldn't find CVC4 where you pointed CVC4_DIR: ${CVC4_DIR}")
    endif ()
endif ()

if ((LIBCVC4 STREQUAL "LIBCVC4-NOTFOUND") AND (NOT $ENV{SATDIR64} STREQUAL ""))
    findcvc4(CVC4VERSIONSTR LIBCVC4 INCCVC4 "$ENV{SATDIR64}/cvc4")
endif ()

if ((LIBCVC4 STREQUAL "LIBCVC4-NOTFOUND") AND (NOT $ENV{SATDIR32} STREQUAL ""))
    findcvc4(CVC4VERSIONSTR LIBCVC4 INCCVC4 "$ENV{SATDIR32}/cvc4")
endif ()

if ((LIBCVC4 STREQUAL LIBCVC4 INCCVC4 "LIBCVC4-NOTFOUND") AND (NOT $ENV{SATDIR} STREQUAL ""))
    findcvc4(CVC4VERSIONSTR LIBCVC4 INCCVC4 "$ENV{SATDIR}/cvc4")
endif ()

if (LIBCVC4 STREQUAL "LIBCVC4-NOTFOUND")
    findcvc4(CVC4VERSIONSTR LIBCVC4 INCCVC4 "$ENV{HOME}/cvc4")
endif ()

if (LIBCVC4 STREQUAL "LIBCVC4-NOTFOUND")
    # Try default search instead
    find_library(LIBCVC4 cvc4)
    set(CVC4VERSIONSTR "(system)")
endif ()

if (LIBCVC4 STREQUAL "LIBCVC4-NOTFOUND")
    if (DEFINED ENABLE_CVC4)
        message(SEND_ERROR "CVC4 requested, but couldn't find it; please point CVC4_DIR at it")
    endif ()
    # Otherwise, CVC4 neither found nor requested, so ignore it.
else ()
    # Got it; build stuff.
    add_library(solvercvc4 cvc_conv.cpp)
    if(CVC4VERSIONSTR STREQUAL "(system)")
        target_include_directories(solvercvc4            
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    else()
        target_include_directories(solvercvc4
            PRIVATE ${INCCVC4}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    endif()   
    
    find_library(gmp gmp)
    if (NOT gmp)
        message(FATAL_ERROR "gmp not found")
    endif ()
    target_link_libraries(solvercvc4 "${LIBCVC4}" ${gmp})
    set(ENABLE_CVC4 "1")
    message(STATUS "CVC4 version: ${CVC4VERSIONSTR}")

    # Add to solver link
    target_link_libraries(solvers INTERFACE solvercvc4)

    set(ESBMC_ENABLE_cvc4 1 PARENT_SCOPE)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} cvc4" PARENT_SCOPE)
endif ()
