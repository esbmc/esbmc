set(CVC4_MIN_VERSION "1.6")

if(DEFINED CVC5_DIR)
    set(ENABLE_CVC5 ON)
endif()

if(EXISTS $ENV{HOME}/cvc5)
    set(ENABLE_CVC5 ON)
endif()

if(ENABLE_CVC5)
    find_package(CVC5 REQUIRED
            HINTS ${CVC4_DIR}/lib/cmake/CVC5 $ENV{HOME}/cvc5)

    message(STATUS "Found CVC5 at: ${CVC5_DIR}")
    message(STATUS "CVC4 version: ${CVC5_VERSION}")
    if(CVC5_VERSION VERSION_LESS CVC5_MIN_VERSION)
        message(FATAL_ERROR "Expected version ${CVC4_MIN_VERSION} or greater")
    endif()

    add_library(solvercvc5 cvc_conv.cpp)
    target_include_directories(solvercvc5
            PRIVATE ${Boost_INCLUDE_DIRS}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    find_library(gmp gmp)
    if (NOT gmp)
        message(FATAL_ERROR "gmp not found")
    endif ()
    target_link_libraries(solvercvc5 fmt::fmt "cvc5::cvc5" ${gmp})

    # Add to solver link
    target_link_libraries(solvers INTERFACE solvercvc5)

    set(ESBMC_ENABLE_cvc5 1 PARENT_SCOPE)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} cvc5" PARENT_SCOPE)
endif()
