set(CVC4_MIN_VERSION "1.6")

if(DEFINED CVC4_DIR)
    set(ENABLE_CVC4 ON)
endif()

if(EXISTS $ENV{HOME}/cvc4)
    set(ENABLE_CVC4 ON)
endif()

if(ENABLE_CVC4)
    find_library(CVC4_LIB cvc4 PATHS ${CVC4_DIR} $ENV{HOME}/cvc4 PATH_SUFFIXES lib bin)
    find_path(CVC4_INCLUDE_DIRS cvc4/cvc4.h PATHS ${CVC4_DIR} $ENV{HOME}/cvc4 PATH_SUFFIXES include)

    if(CVC4_INCLUDE_DIRS STREQUAL "CVC4_INCLUDE_DIRS-NOTFOUND")
        message(FATAL_ERROR "Could not find cvc4 include headers, please check CVC4_DIR")
    endif()

    if(CVC4_LIB STREQUAL "CVC4_LIB-NOTFOUND")
        message(FATAL_ERROR "Could not find libcvc4, please check CVC4_DIR")
    endif()

    try_run(CVC4_RUNS CVC4_COMPILES ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/try_cvc4.cpp
            CMAKE_FLAGS -DINCLUDE_DIRECTORIES=${CVC4_INCLUDE_DIRS}
            LINK_LIBRARIES ${CVC4_LIB} gmp
            COMPILE_OUTPUT_VARIABLE CVC4COMPILESTR
            RUN_OUTPUT_VARIABLE CVC4_VERSION)

    if(NOT CVC4_COMPILES)
        message(FATAL_ERROR "Could not use CVC4: \n${CVC4COMPILESTR}")
    endif()

    message(STATUS "Using CVC4 at: ${CVC4_LIB}")
    message(STATUS "CVC4 version: ${CVC4_VERSION}")
    if(CVC4_VERSION VERSION_LESS CVC4_MIN_VERSION)
        message(FATAL_ERROR "Expected version ${CVC4_MIN_VERSION} or greater")
    endif()
    


    add_library(solvercvc4 cvc_conv.cpp)
    target_include_directories(solvercvc4
            PRIVATE ${CVC4_INCLUDE_DIRS}
            PRIVATE ${Boost_INCLUDE_DIRS}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    find_library(gmp gmp)
    if (NOT gmp)
        message(FATAL_ERROR "gmp not found")
    endif ()
    target_link_libraries(solvercvc4 "${CVC4_LIB}" ${gmp})

    # Add to solver link
    target_link_libraries(solvers INTERFACE solvercvc4)
    set(ESBMC_ENABLE_cvc4 1 PARENT_SCOPE)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} cvc4" PARENT_SCOPE)
endif()
