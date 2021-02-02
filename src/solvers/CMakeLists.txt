set (ESBMC_ENABLE_z3 0)
set (ESBMC_ENABLE_vampire 0)
set (ESBMC_ENABLE_minisat 0)
set (ESBMC_ENABLE_boolector 0)
set (ESBMC_ENABLE_cvc4 0)
set (ESBMC_ENABLE_mathsat 0)
set (ESBMC_ENABLE_yices 0)

add_subdirectory(prop)
add_subdirectory(smt)
add_subdirectory(smtlib)

add_library(solve solve.cpp)
target_include_directories(solve
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${Boost_INCLUDE_DIRS}
)

add_library(solvers INTERFACE)
target_link_libraries(solvers INTERFACE solve smtlib smttuple smtfp smt prop)

# Logic for each of these are duplicated -- cmake doesn't have indirect function
# calling, so it's hard to structure it how I want
add_subdirectory(z3)
add_subdirectory(vampire)
add_subdirectory(boolector)
add_subdirectory(cvc4)
add_subdirectory(mathsat)
add_subdirectory(yices)

set(ESBMC_AVAILABLE_SOLVERS "${ESBMC_AVAILABLE_SOLVERS}" PARENT_SCOPE)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/solver_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/solver_config.h")
