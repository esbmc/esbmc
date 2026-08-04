add_library(smttuple smt_tuple_node.cpp smt_tuple_node_ast.cpp)
target_include_directories(smttuple
    PRIVATE ${Boost_INCLUDE_DIRS}
)
target_link_libraries(smttuple PRIVATE fmt::fmt)