add_library(solidityfrontend_stuff
  solidity_language.cpp
  typecast.cpp
  solidity_convert.cpp
  solidity_convert_decl.cpp
  solidity_convert_inheritance.cpp
  solidity_convert_constructor.cpp
  solidity_convert_modifier.cpp
  solidity_convert_stmt.cpp
  solidity_convert_expr.cpp
  solidity_convert_ref.cpp
  solidity_convert_type.cpp
  solidity_convert_tuple.cpp
  solidity_convert_mapping.cpp
  solidity_convert_contract.cpp
  solidity_convert_builtin.cpp
  solidity_convert_call.cpp
  solidity_convert_util.cpp
  solidity_convert_literals.cpp
  solidity_grammar.cpp
  pattern_check.cpp
)
target_include_directories(solidityfrontend_stuff
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE ${CLANG_INCLUDE_DIRS}
    PRIVATE ${Boost_INCLUDE_DIRS}
)

target_link_libraries(solidityfrontend_stuff gotoprograms nlohmann_json::nlohmann_json clangcfrontend_stuff)

add_library(solidityfrontend INTERFACE)
target_link_libraries(solidityfrontend INTERFACE solidityfrontend_stuff)
