if(ENABLE_BOOLECTOR)
    find_package(Boolector REQUIRED
            PATHS ${Boolector_DIR}/lib/cmake)

    message(STATUS "Boolector version: ${Boolector_VERSION}")

    add_library(solverbtor boolector_conv.cpp)
    target_include_directories(solverbtor
            PRIVATE ${Boost_INCLUDE_DIRS}
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(solverbtor Boolector::boolector)

    target_link_libraries(solvers INTERFACE solverbtor)
    set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS} boolector" PARENT_SCOPE)
endif()