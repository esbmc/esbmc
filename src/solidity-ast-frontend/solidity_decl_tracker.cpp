#include <solidity-ast-frontend/solidity_decl_tracker.h>

void decl_function_tracker::config()
{
  //TODO: these configurations were created heavily influenced by clang-frontend.
  //      Some of them may be redundant for Solidity. Future audit might be needed.

  set_isImplicit();
  set_isDefined();
  set_isThisDeclarationADefinition();
  set_type_class();
}

void decl_function_tracker::set_isImplicit()
{
  if (!decl_func.contains("isImplicit"))
    assert(!"missing \'isImplicit\' in DeclFunction");
  isImplicit = (decl_func["isImplicit"].get<bool>())? true : false;
}

void decl_function_tracker::set_isDefined()
{
  if (!decl_func.contains("isDefined"))
    assert(!"missing \'isDefined\' in DeclFunction");
  isDefined = (decl_func["isDefined"].get<bool>())? true : false;
}

void decl_function_tracker::set_isThisDeclarationADefinition()
{
  if (!decl_func.contains("isThisDeclarationADefinition"))
    assert(!"missing \'isThisDeclarationADefinition\' in DeclFunction");
  isThisDeclarationADefinition = (decl_func["isThisDeclarationADefinition"].get<bool>())? true : false;
}

void decl_function_tracker::set_type_class()
{
  if (!decl_func.contains("typeClass"))
    assert(!"missing \'typeClass\' in DeclFunction");
  type_class = SolidityTypes::get_type_class(
      decl_func["typeClass"].get<std::string>()
      );
}

void decl_function_tracker::set_decl_class()
{
  if (!decl_func.contains("declClass"))
    assert(!"missing \'declClass\' in DeclFunction");
  decl_class = SolidityTypes::get_decl_class(
      decl_func["declClass"].get<std::string>()
      );
}

void decl_function_tracker::print_decl_func_json()
{
  const nlohmann::json &json_in = decl_func;
  printf("### decl_func_json: ###\n");
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}

