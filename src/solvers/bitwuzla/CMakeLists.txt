set(Bitwuzla_MIN_VERSION "0.1")

if(DEFINED Bitwuzla_DIR)
    set(ENABLE_BITWUZLA ON)
endif()

if(EXISTS $ENV{HOME}/bitwuzla)
    set(ENABLE_BITWUZLA ON)
endif()

if(ENABLE_BITWUZLA)
    find_package(Bitwuzla REQUIRED
            HINTS ${Bitwuzla_DIR}/lib/cmake $ENV{HOME}/bitwuzla)

    message(STATUS "Found Bitwuzla at: ${Bitwuzla_DIR}")
    message(STATUS "Bitwuzla version: ${Bitwuzla_VERSION}")
    if(Bitwuzla_VERSION VERSION_LESS Bitwuzla_MIN_VERSION)
        message(FATAL_ERROR "Expected version ${Bitwuzla_MIN_VERSION} or greater")
    endif()
   
    add_library(solverbitw bitwuzla_conv.cpp)
    target_include_directories(solverbitw
            PRIVATE ${Boost_INCLUDE_DIRS}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(solverbitw fmt::fmt Bitwuzla::bitwuzla)

    target_link_libraries(solvers INTERFACE solverbitw)

    set(ESBMC_ENABLE_bitwuzla 1 PARENT_SCOPE)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} bitwuzla" PARENT_SCOPE)
endif()
