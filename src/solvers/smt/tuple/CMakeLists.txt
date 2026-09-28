add_library(smttuple smt_tuple.cpp smt_tuple_array_ast.cpp smt_tuple_node.cpp smt_tuple_sym.cpp smt_tuple_node_ast.cpp smt_tuple_sym_ast.cpp)
target_link_libraries(smttuple PUBLIC esbmc_common PRIVATE fmt::fmt)
