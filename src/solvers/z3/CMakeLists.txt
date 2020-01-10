set(Z3_MIN_VERSION "4.8.4")

if(DEFINED Z3_DIR)
    set(ENABLE_Z3 ON)
endif()

if(EXISTS $ENV{HOME}/z3)
    set(ENABLE_Z3 ON)
endif()


if(ENABLE_Z3)
    find_library(Z3_LIB z3 PATHS ${Z3_DIR} $ENV{HOME}/z3 PATH_SUFFIXES lib bin)
    find_path(Z3_INCLUDE_DIRS z3.h PATHS ${Z3_DIR} $ENV{HOME}/z3 PATH_SUFFIXES include)

    if(Z3_INCLUDE_DIRS STREQUAL "Z3_INCLUDE_DIRS-NOTFOUND")
        message(FATAL_ERROR "Could not find z3 include headers, please check Z3_DIR")
    endif()

    if(Z3_LIB STREQUAL "Z3_LIB-NOTFOUND")
        message(FATAL_ERROR "Could not find libz3, please check Z3_DIR")
    endif()

    try_run(Z3_RUNS Z3_COMPILES ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/try_z3.c
            CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${Z3_INCLUDE_DIRS}
            LINK_LIBRARIES ${Z3_LIB} stdc++ m -pthread
            COMPILE_OUTPUT_VARIABLE Z3COMPILESTR
            RUN_OUTPUT_VARIABLE Z3_VERSION)

    if(NOT Z3_COMPILES)
        message(FATAL_ERROR "Could not use Z3: \n${Z3COMPILESTR}")
    endif()

    message(STATUS "Using Z3 at: ${Z3_LIB}")
    string(REGEX MATCH "Z3 ([0-9].[0-9].[0-9].[0-9])"  REGEX_OUTPUT ${Z3_VERSION})
    set(Z3_VERSION "${CMAKE_MATCH_1}")    
    message(STATUS "Z3 version: ${Z3_VERSION}")
    if(Z3_VERSION VERSION_LESS Z3_MIN_VERSION)
        message(FATAL_ERROR "Expected version ${Z3_MIN_VERSION} or greater")
    endif()
    

    add_library(solverz3 z3_conv.cpp)
    target_include_directories(solverz3
            PRIVATE ${Z3_INCLUDE_DIRS}
            PRIVATE ${Boost_INCLUDE_DIRS}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    if (BUILD_STATIC)
        target_link_libraries(solverz3 "${Z3_LIB}" -pthread "${LIBGOMP_LIB}" -ldl)
    else ()
        target_link_libraries(solverz3 "${Z3_LIB}")
    endif ()
    # Add to solver link
    target_link_libraries(solvers INTERFACE solverz3)
    
    set(ESBMC_ENABLE_z3 1 PARENT_SCOPE)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} z3" PARENT_SCOPE)
endif()
