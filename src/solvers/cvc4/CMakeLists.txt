set(CVC4_MIN_VERSION "1.6")

if(DEFINED CVC4_DIR)
    set(ENABLE_CVC4 ON PARENT_SCOPE)

elseif(EXISTS $ENV{HOME}/cvc4)
    set(ENABLE_CVC4 ON PARENT_SCOPE)

else()
    if(ENABLE_CVC4 AND DOWNLOAD_DEPENDENCIES)
        # TODO: Replace this file when we go for CVC5!
        CPMAddPackage(
            NAME cvc5
            DOWNLOAD_ONLY YES
            GITHUB_REPOSITORY cvc5/cvc5
            GIT_TAG cvc5-1.0.5)

        message("[cvc5] Configuring")
        execute_process(COMMAND ./configure.sh --auto-download --prefix=${cvc5_BINARY_DIR}
                            WORKING_DIRECTORY ${cvc5_SOURCE_DIR})

        message("[cvc5] Building")
        execute_process(COMMAND make -j4 install
            WORKING_DIRECTORY ${cvc5_SOURCE_DIR}/build)

        set(CVC4_DIR ${cvc5_BINARY_DIR})
    endif()
endif()


if(ENABLE_CVC4)
    find_package(CVC4 REQUIRED
            HINTS ${CVC4_DIR}/lib/cmake/CVC4 $ENV{HOME}/cvc4)

    message(STATUS "Found CVC4 at: ${CVC4_DIR}")
    message(STATUS "CVC4 version: ${CVC4_VERSION}")
    if(CVC4_VERSION VERSION_LESS CVC4_MIN_VERSION)
        message(FATAL_ERROR "Expected version ${CVC4_MIN_VERSION} or greater")
    endif()

    add_library(solvercvc4 cvc_conv.cpp)
    target_include_directories(solvercvc4
            PRIVATE ${Boost_INCLUDE_DIRS}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(solvercvc4 fmt::fmt "CVC4::cvc4")

    # Add to solver link
    target_link_libraries(solvers INTERFACE solvercvc4)

    set(ESBMC_ENABLE_cvc4 1 PARENT_SCOPE)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} cvc4" PARENT_SCOPE)
endif()
