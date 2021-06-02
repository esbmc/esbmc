#include <solidity-ast-frontend/solidity_convert.h>
#include <util/arith_tools.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/i2string.h>
#include <util/mp_arith.h>
#include <util/std_code.h>
#include <util/std_expr.h>

solidity_convertert::solidity_convertert(
  contextt &_context,
  nlohmann::json &_ast_json
  )
  : context(_context),
    ast_json(_ast_json)
{
}

bool solidity_convertert::convert()
{
  if(convert_top_level_decl())
    return true;

  return false;
}

bool solidity_convertert::convert_top_level_decl()
{
  assert(!"cool ...");

  return false;
}
