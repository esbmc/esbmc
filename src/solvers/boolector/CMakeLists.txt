function(findboolector versionstr libbtor incbtor liblgl thepath libbtor2parser)
    find_library(LIBBTOR boolector NAMES libboolector.a PATHS "${thepath}" PATH_SUFFIXES build/lib NO_DEFAULT_PATH)
    find_path(INCBTOR boolector.h PATHS "${thepath}" PATH_SUFFIXES src NO_DEFAULT_PATH)
    find_path(INCBTORBUILD btorconfig.h PATHS "${thepath}" PATH_SUFFIXES build/src NO_DEFAULT_PATH)

    if ((LIBBTOR STREQUAL "LIBBTOR-NOTFOUND") OR (INCBTOR STREQUAL "INCBTOR-NOTFOUND"))
        return()
    endif ()

    find_library(LIBLGL lgl NAMES liblgl.a PATHS "${thepath}" PATH_SUFFIXES deps/lingeling NO_DEFAULT_PATH)
    find_library(LIBBTORPARSER NAMES libbtor2parser.a PATHS "${thepath}" PATH_SUFFIXES deps/btor2tools/build NO_DEFAULT_PATH)
    get_filename_component(BTOR_LIBDIR "${LIBBTOR}" DIRECTORY)
    get_filename_component(BTOR_INCDIR "${INCBTOR}" DIRECTORY)

    find_program(BOOLECTOR_BIN boolector_bin NAMES boolector PATHS "${thepath}" PATH_SUFFIXES build/bin NO_DEFAULT_PATH)
    set(EXECUTE_BOOLECTOR_VERSION ${BOOLECTOR_BIN} --version)
    execute_process(COMMAND ${EXECUTE_BOOLECTOR_VERSION} OUTPUT_VARIABLE BTORVERSIONSTR OUTPUT_STRIP_TRAILING_WHITESPACE)

    set(${libbtor} "${LIBBTOR}" PARENT_SCOPE)
    set(${liblgl} "${LIBLGL}" PARENT_SCOPE)
    set(${incbtor} "${BTOR_INCDIR}" PARENT_SCOPE)
    set(${versionstr} "${BTORVERSIONSTR}" PARENT_SCOPE)
    set(${libbtor2parser} "${LIBBTORPARSER}" PARENT_SCOPE)
endfunction()

if (DEFINED DISABLE_BTOR)
    if (DEFINED ENABLE_BTOR)
        message(SEND_ERROR "Can't request both disable and enable boolector")
    endif ()
    set(YESWEUSED_DIABLE_BTOR, "${DISABLE_BTOR}")
    return()
endif ()

set(LIBBTOR "LIBBTOR-NOTFOUND")
set(INCBTOR "LIBBTOR-NOTFOUND")
set(LIBBTOR2PARSER "LIBBTOR-NOTFOUND")

if (DEFINED BTOR_DIR)
    findboolector(BTORVERSIONSTR LIBBTOR INCBTOR LIBLGL "${BTOR_DIR}" LIBBTOR2PARSER)
    if (LIBBTOR STREQUAL "LIBBTOR-NOTFOUND")
        message(SEND_ERROR "Couldn't find boolector where you pointed BTOR_DIR: ${BTOR_DIR}")
    endif ()
endif ()

if ((LIBBTOR STREQUAL "LIBBTOR-NOTFOUND") AND (NOT $ENV{SATDIR64} STREQUAL ""))
    findboolector(BTORVERSIONSTR LIBBTOR INCBTOR LIBLGL "$ENV{SATDIR64}/boolector" LIBBTOR2PARSER)
endif ()

if ((LIBBTOR STREQUAL "LIBBTOR-NOTFOUND") AND (NOT $ENV{SATDIR32} STREQUAL ""))
    findboolector(BTORVERSIONSTR LIBBTOR INCBTOR LIBLGL "$ENV{SATDIR32}/boolector" LIBBTOR2PARSER)
endif ()

if ((LIBBTOR STREQUAL LIBBTOR INCBTOR "LIBBTOR-NOTFOUND") AND (NOT $ENV{SATDIR} STREQUAL ""))
    findboolector(BTORVERSIONSTR LIBBTOR INCBTOR LIBLGL "$ENV{SATDIR}/boolector" LIBBTOR2PARSER)
endif ()

if (LIBBTOR STREQUAL "LIBBTOR-NOTFOUND")
    findboolector(BTORVERSIONSTR LIBBTOR INCBTOR LIBLGL "$ENV{HOME}/boolector" LIBBTOR2PARSER)
endif ()

if (LIBBTOR STREQUAL "LIBBTOR-NOTFOUND")
    # Try default search instead
    find_library(LIBBTOR boolector)
    set(BTORVERSIONSTR "(system)")
endif ()

if (LIBBTOR STREQUAL "LIBBTOR-NOTFOUND")
    if (DEFINED ENABLE_BTOR)
        message(SEND_ERROR "boolector requested, but couldn't find it; please point BTOR_DIR at it")
    endif ()
    # Otherwise, boolector neither found nor requested, so ignore it.
else ()
    # Got it; build stuff.    
    add_library(solverbtor boolector_conv.cpp)
    target_include_directories(solverbtor
            PRIVATE ${INCBTOR}/src
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(solverbtor ${LIBBTOR} ${LIBLGL} ${LIBBTOR2PARSER})
    set(ENABLE_BTOR "1")
    message(STATUS "Boolector version: ${BTORVERSIONSTR}")

    # Add to solver link
    target_link_libraries(solvers INTERFACE solverbtor)

    set(ESBMC_ENABLE_boolector 1 PARENT_SCOPE)
endif ()