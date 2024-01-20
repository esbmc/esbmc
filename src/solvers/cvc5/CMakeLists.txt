if(EXISTS $ENV{HOME}/cvc5)
  set(ENABLE_CVC5 ON)

else()
  if(ENABLE_CVC5 AND DOWNLOAD_DEPENDENCIES)
        CPMAddPackage(
            NAME cvc5
            DOWNLOAD_ONLY YES
            GITHUB_REPOSITORY cvc5/cvc5
            GIT_TAG cvc5-1.1.0)

	# Require python pyparsing!
        message("[cvc5] Configuring")
        execute_process(COMMAND ./configure.sh --auto-download --prefix=${cvc5_BINARY_DIR}
                            WORKING_DIRECTORY ${cvc5_SOURCE_DIR})

        message("[cvc5] Building")
        execute_process(COMMAND make -j4 install
            WORKING_DIRECTORY ${cvc5_SOURCE_DIR}/build)

        set(CVC5_DIR ${cvc5_BINARY_DIR})
    endif()
endif()

if(ENABLE_CVC5)
    message(STATUS "Looking for CVC5 at: ${cvc5_BINARY_DIR}") 
    find_package(cvc5 REQUIRED
            HINTS ${CVC5_DIR}/lib/cmake ${CVC5_DIR}/lib64/cmake $ENV{HOME}/cvc5)

    message(STATUS "Found CVC5 at: ${CVC5_DIR}") 
    message(STATUS "CVC5 version: ${cvc5_VERSION}")

    add_library(solvercvc5 cvc_conv.cpp)
    target_include_directories(solvercvc5
            PRIVATE ${Boost_INCLUDE_DIRS}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(solvercvc5 fmt::fmt "cvc5::cvc5")

    # Add to solver link
    target_link_libraries(solvers INTERFACE solvercvc5)

    set(ESBMC_ENABLE_cvc5 1 PARENT_SCOPE)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} cvc5" PARENT_SCOPE)
endif()
