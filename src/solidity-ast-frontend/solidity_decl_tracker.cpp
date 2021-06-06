#include <solidity-ast-frontend/solidity_decl_tracker.h>

void decl_function_tracker::config()
{
  //TODO: these configurations were created heavily influenced by clang-frontend.
  //      Some of them may be redundant for Solidity. Future audit might be needed.

  set_isImplicit();
}

void decl_function_tracker::print_decl_func_json()
{
  const nlohmann::json &json_in = decl_func;
  printf("### decl_func_json: ###\n");
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}

void decl_function_tracker::set_isImplicit()
{
  assert(!"good!!!");
}
