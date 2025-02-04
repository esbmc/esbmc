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

template <typename T>
auto create_list(int size, T default_value)
{
  nlohmann::json list;
  list["_type"] = "List";
  for (int i = 0; i < size; ++i)
  {
    list["elts"].push_back({{"_type", "Constant"}, {"value", default_value}});
  }
  return list;
}

exprt numpy_call_expr::get()
{
  static const std::unordered_map<std::string, float> numpy_functions = {
    {"zeros", 0.0}, {"ones", 1.0}};

  const std::string &function = function_id_.get_function();

  if (function == "array")
  {
    return converter_.get_expr(call_["args"][0]);
  }

  auto it = numpy_functions.find(function);
  if (it != numpy_functions.end())
  {
    auto list = create_list(call_["args"][0]["value"].get<int>(), it->second);
    return converter_.get_expr(list);
  }

  throw std::runtime_error("Unsupported NumPy function call: " + function);
}
