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

solidity_convertert::solidity_convertert(
  contextt &_context,
  nlohmann::json &_ast_json
  )
  : context(_context),
    ast_json(_ast_json)
{
}

// This method convert declarations. They are called when those declarations
// are to be added to the context. If a variable or function is being called
// but then get_decl_expr is called instead
bool solidity_convertert::convert()
{
  // Iterate through each intrinsic nodes and AST nodes, creating
  // symbols as we go.
  nlohmann::json::iterator it = ast_json.begin();
  unsigned index = 0;
  for (; it != ast_json.end(); ++it, ++index)
  {
    print_json_element(*it, index, it.key());
  }

  assert(!"cool");

  return false;
}

void solidity_convertert::print_json_element(const nlohmann::json &json_in, const unsigned index, const std::string &key)
{
  printf("\n### json element[%u] content: key=%s, size=%lu ###\n", index, key.c_str(), json_in.size());
  std::cout << std::setw(2) << json_in << '\n'; // '2' means 2x indentations
  printf("\n");
}

