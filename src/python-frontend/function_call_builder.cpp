#include <python-frontend/function_call_builder.h>
#include <python-frontend/function_call_expr.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/json_utils.h>

function_call_builder::function_call_builder(python_converter &converter)
  : converter_(converter)
{
}

bool is_numpy_call(const symbol_id &function_id)
{
  const std::string &filename = function_id.get_filename();
  const std::string &suffix = "/models/numpy.py";

  return (filename.rfind(suffix) == (filename.size() - suffix.size()));
}

exprt function_call_builder::build(const nlohmann::json &call) const
{
  function_call_expr call_expr(call, converter_);

  const symbol_id &function_id = call_expr.get_function_id();

  // Handle NumPy functions
  if (is_numpy_call(function_id) && function_id.get_function() == "array")
  {
    // Get array from function arguments
    // TODO: Add support for multidimensional arrays
    exprt array = converter_.get_expr(call["args"][0]);
    return array;
  }

  return call_expr.get();
}
