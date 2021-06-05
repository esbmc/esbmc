#include <solidity-ast-frontend/solidity_convert.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/mp_arith.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <iomanip>

solidity_convertert::solidity_convertert(contextt &_context, nlohmann::json &_ast_json, nlohmann::json &_intrinsic_json):
  context(_context),
  ast_json(_ast_json),
  intrinsic_json(_intrinsic_json)
{
}

bool solidity_convertert::convert()
{
  // This method convert each declarations in ast_json to symbols and add them to the context.
  // The conversion consists of three parts:
  //  - PART 1: get declarations from intrinsics include ESBMC and TACAS definitions
  //  - PART 2: get declarations from AST nodes

  // PART 1: get declarations from intrinsics include ESBMC and TACAS definitions
  unsigned index = 0;
  for (auto it = intrinsic_json.begin(); it != intrinsic_json.end(); ++it, ++index)
  {
    // iterate over each json object (i.e. ESBMC or TACAS definitions) and add symbols as we go
    const std::string& top_level_key = it.key();
    printf("@@ current key=%s\n", top_level_key.c_str());
    exprt dummy_decl;
    get_decl_intrinsics(it.value(), dummy_decl, index, top_level_key, "intrinsic_json");
  }

  //TODO:  - PART 2: get declarations from AST nodes
  assert(!"cool");

  return false;
}


bool solidity_convertert::get_decl_intrinsics(
    const nlohmann::json& decl, exprt &new_expr,
    const unsigned index, const std::string &key, const std::string &json_name)
{
  // This method convert declarations from intrinsics. They are called when those declarations
  // are to be added to the context.
  printf("In function %s: \n", __func__);
  print_json_element(decl, index, key, json_name);

  return false;
}

void solidity_convertert::print_json_element(const nlohmann::json &json_in, const unsigned index,
    const std::string &key, const std::string& json_name)
{
  printf("\n### %s element[%u] content: key=%s, size=%lu ###\n",
      json_name.c_str(), index, key.c_str(), json_in.size());
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations in front of each line
  printf("\n");
}
