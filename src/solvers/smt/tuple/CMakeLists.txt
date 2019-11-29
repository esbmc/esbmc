add_library(smttuple smt_tuple_array_ast.cpp smt_tuple_node.cpp smt_tuple_sym.cpp smt_tuple_node_ast.cpp smt_tuple_sym_ast.cpp)
target_include_directories(smttuple
    PRIVATE ${Boost_INCLUDE_DIRS}
)