set(CVC4_MIN_VERSION "1.6")

if(DEFINED CVC4_DIR)
    set(ENABLE_CVC4 ON)
endif()

if(EXISTS $ENV{HOME}/cvc4)
    set(ENABLE_CVC4 ON)
endif()

if(ENABLE_CVC4)
    find_package(CVC4 REQUIRED
            PATHS ${CVC4_DIR}/lib/cmake/CVC4 $ENV{HOME}/cvc4)

    message(STATUS "Found CVC4 at: ${CVC4_DIR}")
    message(STATUS "CVC4 version: ${CVC4_VERSION}")
    if(CVC4_VERSION VERSION_LESS CVC4_MIN_VERSION)
        message(FATAL_ERROR "Expected version ${CVC4_MIN_VERSION} or greater")
    endif()

    add_library(solvercvc4 cvc_conv.cpp)
    target_include_directories(solvercvc4
            PRIVATE ${Boost_INCLUDE_DIRS}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    find_library(gmp gmp)
    if (NOT gmp)
        message(FATAL_ERROR "gmp not found")
    endif ()
    target_link_libraries(solvercvc4 "CVC4::cvc4" ${gmp})

    # Add to solver link
    target_link_libraries(solvers INTERFACE solvercvc4)

    set(ESBMC_ENABLE_cvc4 1 PARENT_SCOPE)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} cvc4" PARENT_SCOPE)
endif()
