set(ESBMC_ENABLE_z3 0)
set(ESBMC_ENABLE_minisat 0)
set(ESBMC_ENABLE_cvc5 0)
set(ESBMC_ENABLE_mathsat 0)
set(ESBMC_ENABLE_yices 0)
set(ESBMC_ENABLE_bitwuzla 0)
set(ESBMC_ENABLE_smtlib 0)

add_subdirectory(prop)
add_subdirectory(smt)


add_library(solve solve.cpp)
target_link_libraries(solve PUBLIC smt PRIVATE fmt::fmt)
target_include_directories(solve
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${Boost_INCLUDE_DIRS}
)

add_library(solvers INTERFACE)
target_link_libraries(solvers INTERFACE solve smt prop)

add_subdirectory(camada)

set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS}" PARENT_SCOPE)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/solver_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/solver_config.h")
