set (ESBMC_ENABLE_z3 0)
set (ESBMC_ENABLE_minisat 0)
set (ESBMC_ENABLE_boolector 0)
set (ESBMC_ENABLE_cvc4 0)
set (ESBMC_ENABLE_mathsat 0)
set (ESBMC_ENABLE_yices 0)

add_subdirectory(prop)
add_subdirectory(smt)
add_subdirectory(smtlib)

add_library(solve solve.cpp smt_python.cpp)
target_include_directories(solve     
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
    PRIVATE ${Boost_INCLUDE_DIRS}
)
if (DEFINED HAVE_PYTHON)
  target_link_libraries(solve ${PYTHON_LINK_LIBS})
endif()

add_library(solvers INTERFACE)
target_link_libraries(solvers INTERFACE solve smtlib smttuple smtfp smt prop)

# Logic for each of these are duplicated -- cmake doesn't have indirect function
# calling, so it's hard to structure it how I want
#add_subdirectory(z3)
#add_subdirectory(boolector)
#add_subdirectory(cvc4)
#add_subdirectory(mathsat)
#add_subdirectory(yices)

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/solver_config.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/solver_config.h")
