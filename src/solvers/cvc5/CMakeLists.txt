set(CVC5_MIN_VERSION "1.0.2")

if(DEFINED CVC5_DIR)
    set(ENABLE_CVC5 ON)
endif()

if(EXISTS $ENV{HOME}/cvc5)
  set(ENABLE_CVC5 ON)
else()
  if(ENABLE_CVC5 AND DOWNLOAD_DEPENDENCIES)
        download_zip_and_extract(CVC5 ${ESBMC_CVC5_URL})
        set(CVC5_DIR ${CMAKE_BINARY_DIR}/CVC5/${ESBMC_CVC5_NAME})
    endif()
endif()

if(ENABLE_CVC5)
    message(STATUS "Looking for CVC5 at: ${CVC5_DIR}") 
    find_package(cvc5 REQUIRED
            HINTS ${CVC5_DIR}/lib/cmake ${CVC5_DIR}/lib64/cmake $ENV{HOME}/cvc5)
    message(STATUS "Found CVC5 at: ${cvc5_DIR}")
    message(STATUS "CVC5 version: ${cvc5_VERSION}")
    if(cvc5_VERSION VERSION_LESS CVC5_MIN_VERSION)
        message(FATAL_ERROR "Expected version ${CVC5_MIN_VERSION} or greater")
    endif()

    add_library(solvercvc5 cvc5_conv.cpp)
    target_include_directories(solvercvc5
            PRIVATE ${Boost_INCLUDE_DIRS}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(solvercvc5 PUBLIC fmt::fmt cvc5::cvc5)

    # Add to solver link
    target_link_libraries(solvers INTERFACE solvercvc5)

    set(ESBMC_ENABLE_cvc5 1 PARENT_SCOPE)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} cvc5" PARENT_SCOPE)
endif()
