set(Yices_MIN_VERSION "2.6.1")

if(DEFINED Yices_DIR)
    set(ENABLE_YICES ON)
else()
    if(ENABLE_YICES AND DOWNLOAD_DEPENDENCIES)
    # Yices does not support CMake
    CPMAddPackage(
        NAME yices
        DOWNLOAD_ONLY YES
        GITHUB_REPOSITORY SRI-CSL/yices2
        GIT_TAG Yices-2.6.4)

    # We can't use ExternalProject because we need it ready
    # at configuration time

    message("[yices] Preparing")
    execute_process(COMMAND "autoreconf"
                WORKING_DIRECTORY ${yices_SOURCE_DIR})

    message("[yices] Configuring build")
    execute_process(COMMAND ${yices_SOURCE_DIR}/configure --prefix ${yices_BINARY_DIR}
                WORKING_DIRECTORY ${yices_SOURCE_DIR})

    message("[yices] Building...")
    execute_process(COMMAND make -j4
        WORKING_DIRECTORY ${yices_SOURCE_DIR})

    message("[yices] Building...")
    execute_process(COMMAND make static-lib
        WORKING_DIRECTORY ${yices_SOURCE_DIR})

    message("[yices] Installing...")
    execute_process(COMMAND make install
        WORKING_DIRECTORY ${yices_SOURCE_DIR})

    file(COPY ${yices_SOURCE_DIR}/build/x86_64-pc-linux-gnu-release/static_lib/libyices.a 
        DESTINATION ${yices_BINARY_DIR}/lib/)

    set(Yices_DIR ${yices_BINARY_DIR})
endif()

endif()

if(EXISTS $ENV{HOME}/yices)
    set(ENABLE_YICES ON)
endif()

if(ENABLE_YICES)
    find_path(Yices_INCLUDE_DIRS yices.h HINTS "${Yices_DIR}" $ENV{HOME}/yices PATH_SUFFIXES include)
    find_library(Yices_LIB yices NAMES yices HINTS "${Yices_DIR}" $ENV{HOME}/yices PATH_SUFFIXES lib)

    if(Yices_INCLUDE_DIRS STREQUAL "Yices_INCLUDE_DIRS-NOTFOUND")
        message(FATAL_ERROR "Could not find yices headers, please check Yices_DIR")
    endif()

    if(Yices_LIB STREQUAL "Yices_LIB-NOTFOUND")
        message(FATAL_ERROR "Could not find libyices, please check Yices_DIR")
    endif()

    try_run(Yices_RUNS Yices_COMPILES ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/try_yices.c
            CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${Yices_INCLUDE_DIRS}
            LINK_LIBRARIES ${Yices_LIB}
            COMPILE_OUTPUT_VARIABLE YICESCOMPILESTR
            RUN_OUTPUT_VARIABLE Yices_VERSION)

    if(NOT Yices_COMPILES)
        message(FATAL_ERROR "Could not use Yices: \n${YICESCOMPILESTR}")
    endif()

    message(STATUS "Using Yices at: ${Yices_LIB}")
    message(STATUS "Yices version: ${Yices_VERSION}")

    if(Yices_VERSION VERSION_LESS Yices_MIN_VERSION)
        message(FATAL_ERROR "Expected version ${Yices_MIN_VERSION} or greater")
    endif()

    if (Yices_VERSION VERSION_GREATER 2.5.2)
        message(STATUS "Danger Will Robinson: the version of Yices you have is GPL licensed, distribution is impossible")
    endif ()

    add_library(solveryices yices_conv.cpp)

    target_include_directories(solveryices
            PRIVATE ${Yices_INCLUDE_DIRS}
            PRIVATE ${Boost_INCLUDE_DIRS}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    # Hack needed for Ubuntu, since it is not linking with static libs from system
    if(DEFINED GMP_DIR)
        find_library(LIBGMP_CUSTOM gmp NAMES libgmp.a HINTS ${GMP_DIR} PATH_SUFFIXES lib NO_DEFAULT_PATH)
        message(STATUS "Custom gmp found: ${LIBGMP_CUSTOM}")
        target_link_libraries(solveryices fmt::fmt "${Yices_LIB}" "${LIBGMP_CUSTOM}")
    else ()
        target_link_libraries(solveryices fmt::fmt "${Yices_LIB}")
    endif ()

    target_link_libraries(solvers INTERFACE solveryices)
    set(ESBMC_ENABLE_yices 1 PARENT_SCOPE)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} yices" PARENT_SCOPE)
endif()
