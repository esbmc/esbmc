set(Z3_MIN_VERSION "4.8.9")

if(DEFINED Z3_DIR)
    set(ENABLE_Z3 ON)
elseif(EXISTS $ENV{HOME}/z3)
    set(ENABLE_Z3 ON)
else()
    if(ENABLE_Z3 AND DOWNLOAD_DEPENDENCIES)
        message("Downloading Z3")
        download_zip_and_extract(Z3 ${ESBMC_Z3_URL})
        set(Z3_DIR ${CMAKE_BINARY_DIR}/Z3/${ESBMC_Z3_NAME})
    endif()
endif()

if(ENABLE_Z3)
    find_library(Z3_LIB z3 libz3 HINTS ${Z3_DIR} $ENV{HOME}/z3 PATH_SUFFIXES lib bin)
    find_path(Z3_INCLUDE_DIRS z3.h HINTS ${Z3_DIR} $ENV{HOME}/z3 PATH_SUFFIXES include)

    if(Z3_INCLUDE_DIRS STREQUAL "Z3_INCLUDE_DIRS-NOTFOUND")
        message(FATAL_ERROR "Could not find z3 include headers, please check Z3_DIR")
    endif()

    if(Z3_LIB STREQUAL "Z3_LIB-NOTFOUND")
        message(FATAL_ERROR "Could not find libz3, please check Z3_DIR")
    endif()

    try_run(Z3_RUNS Z3_COMPILES ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/try_z3.c
            CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${Z3_INCLUDE_DIRS}
            LINK_LIBRARIES ${Z3_LIB} ${OS_Z3_LIBS}
            COMPILE_OUTPUT_VARIABLE Z3COMPILESTR
            RUN_OUTPUT_VARIABLE Z3_VERSION)

    if(NOT Z3_COMPILES)
        message(FATAL_ERROR "Could not use Z3: \n${Z3COMPILESTR}")
    endif()

    message(STATUS "Using Z3 at: ${Z3_LIB}")
    string(REGEX MATCH "(Z3 )?([0-9]+.[0-9]+.[0-9]+.[0-9]+)"  REGEX_OUTPUT ${Z3_VERSION})
    set(Z3_VERSION "${CMAKE_MATCH_2}")
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
        target_link_libraries(solverz3 fmt::fmt "${Z3_LIB}" -pthread "${LIBGOMP_LIB}" -ldl)
    else ()
        target_link_libraries(solverz3 fmt::fmt "${Z3_LIB}")
    endif ()
    # Add to solver link
    target_link_libraries(solvers INTERFACE solverz3)

    # Install dll for Windows
    if(WIN32)
      find_file(Z3_DLL libz3.dll HINTS ${Z3_DIR} $ENV{HOME}/z3 PATH_SUFFIXES lib bin)
      if(NOT Z3_DLL STREQUAL "Z3_DLL-NOTFOUND")
        message(STATUS "Windows will install ${Z3_DLL}")
        install(FILES ${Z3_DLL} DESTINATION bin)
      endif()
    endif()

    set(ESBMC_ENABLE_z3 1 PARENT_SCOPE)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} z3" PARENT_SCOPE)
endif()
