set(ESBMC_ENABLE_z3 0)
set(ESBMC_ENABLE_minisat 0)
set(ESBMC_ENABLE_boolector 0)
set(ESBMC_ENABLE_cvc4 0)
set(ESBMC_ENABLE_mathsat 0)
set(ESBMC_ENABLE_yices 0)

add_library(
  solve
  solve.cpp
  pointer_logic.cpp
  array_conv.cpp
  smt_byteops.cpp
  smt_casts.cpp
  smt_conv.cpp
  smt_memspace.cpp
  smt_overflow.cpp
  smt_bitcast.cpp
  fp_conv.cpp
  smt_tuple_array_ast.cpp
  smt_tuple_node.cpp
  smt_tuple_sym.cpp
  smt_tuple_node_ast.cpp
  smt_tuple_sym_ast.cpp)

target_include_directories(
  solve
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
  PRIVATE ${Boost_INCLUDE_DIRS})

find_package(camada REQUIRED)

add_library(solvers INTERFACE)
target_link_libraries(solvers INTERFACE solve camada::camada)

set(ESBMC_AVAILABLE_SOLVERS
    "${ESBMC_AVAILABLE_SOLVERS}"
    PARENT_SCOPE)
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/solver_config.h.in"
               "${CMAKE_CURRENT_BINARY_DIR}/solver_config.h")
