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
static auto create_list(int size, T default_value)
{
  nlohmann::json list;
  list["_type"] = "List";
  for (int i = 0; i < size; ++i)
  {
    list["elts"].push_back({{"_type", "Constant"}, {"value", default_value}});
  }
  return list;
}

template <typename T>
static auto create_binary_op(const std::string &op, T lhs, T rhs)
{
  nlohmann::json bin_op = {
    {"_type", "BinOp"},
    {"left", {{"_type", "Constant"}, {"value", lhs}}},
    {"op", {{"_type", op}}},
    {"right", {{"_type", "Constant"}, {"value", rhs}}}};

  return bin_op;
}

bool numpy_call_expr::is_math_function() const
{
  const std::string &function = function_id_.get_function();
  return (function == "add") || (function == "subtract") ||
         (function == "multiply") || (function == "divide") ||
         (function == "power");
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

  if (is_math_function())
  {
    auto bin_op = create_binary_op(
      function, call_["args"][0]["value"], call_["args"][1]["value"]);
    return converter_.get_expr(bin_op);
  }

  throw std::runtime_error("Unsupported NumPy function call: " + function);
}
