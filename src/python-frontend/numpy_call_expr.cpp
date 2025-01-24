#include <python-frontend/numpy_call_expr.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/python_converter.h>
#include <util/expr.h>

numpy_call_expr::numpy_call_expr(
  const symbol_id &function_id,
  const nlohmann::json &call,
  python_converter &converter)
  : function_id_(function_id), call_(call), converter_(converter)
{
}

exprt numpy_call_expr::get()
{
  if (function_id_.get_function() == "array")
  {
    // Get array from function arguments
    // TODO: Add support for multidimensional arrays
    exprt array = converter_.get_expr(call_["args"][0]);
    return array;
  }
  throw std::runtime_error("Unsupported NumPy function call");
}
