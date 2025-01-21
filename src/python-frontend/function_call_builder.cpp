#include <python-frontend/function_call_builder.h>
#include <python-frontend/function_call_expr.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/json_utils.h>

function_call_builder::function_call_builder(python_converter &converter)
  : converter_(converter)
{
}

exprt function_call_builder::build(const nlohmann::json &call) const
{
  function_call_expr call_expr(call, converter_);
  return call_expr.build();
}
