#include <python-frontend/numpy_call_expr.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/python_converter.h>
#include <util/expr.h>
#include <util/c_types.h>
#include <util/message.h>

numpy_call_expr::numpy_call_expr(
  const symbol_id &function_id,
  const nlohmann::json &call,
  python_converter &converter)
  : function_id_(function_id), call_(call), converter_(converter)
{
}

static double extract_value(const nlohmann::json &arg)
{
  if (!arg.contains("_type"))
  {
    throw std::runtime_error("Invalid JSON: missing _type");
  }

  if (arg["_type"] == "UnaryOp")
  {
    if (!arg.contains("operand") || !arg["operand"].contains("value"))
    {
      throw std::runtime_error("Invalid UnaryOp: missing operand/value");
    }
    return -arg["operand"]["value"].get<double>();
  }

  if (!arg.contains("value"))
  {
    throw std::runtime_error("Invalid JSON: missing value");
  }

  return arg["value"].get<double>();
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

std::string numpy_call_expr::get_dtype() const
{
  if (call_.contains("keywords"))
  {
    for (const auto &kw : call_["keywords"])
    {
      if (kw["_type"] == "keyword" && kw["arg"] == "dtype")
      {
        return kw["value"]["attr"];
      }
    }
  }
  return {};
}

size_t numpy_call_expr::get_dtype_size() const
{
  static const std::unordered_map<std::string, size_t> dtype_sizes = {
    {"int8", sizeof(int8_t)},
    {"uint8", sizeof(uint8_t)},
    {"int16", sizeof(int16_t)},
    {"uint16", sizeof(uint16_t)},
    {"int32", sizeof(int32_t)},
    {"uint32", sizeof(uint32_t)},
    {"int64", sizeof(int64_t)},
    {"uint64", sizeof(uint64_t)},
    {"float16", 2},
    {"float32", sizeof(float)},
    {"float64", sizeof(double)}};

  const std::string dtype = get_dtype();
  if (!dtype.empty())
  {
    auto it = dtype_sizes.find(dtype);
    if (it != dtype_sizes.end())
    {
      return it->second * 8;
    }
    throw std::runtime_error("Unsupported dtype value: " + dtype);
  }
  return 0;
}

size_t count_effective_bits(const std::string &binary)
{
  size_t first_one = binary.find('1');
  if (first_one == std::string::npos)
  {
    return 1;
  }
  return binary.size() - first_one;
}

typet numpy_call_expr::get_typet_from_dtype() const
{
  std::string dtype = get_dtype();
  if (dtype.find("int") != std::string::npos)
  {
    if (dtype[0] == 'u')
      return unsignedbv_typet(get_dtype_size());
    return signedbv_typet(get_dtype_size());
  }
  if (dtype.find("float") != std::string::npos)
    return build_float_type(get_dtype_size());

  return {};
}

exprt numpy_call_expr::get() const
{
  static const std::unordered_map<std::string, float> numpy_functions = {
    {"zeros", 0.0}, {"ones", 1.0}};

  const std::string &function = function_id_.get_function();

  // Create array from numpy.array()
  if (function == "array")
  {
    return converter_.get_expr(call_["args"][0]);
  }

  // Create array from numpy.zeros() or numpy.ones()
  auto it = numpy_functions.find(function);
  if (it != numpy_functions.end())
  {
    auto list = create_list(call_["args"][0]["value"].get<int>(), it->second);
    return converter_.get_expr(list);
  }

  // Handle math function calls
  if (is_math_function())
  {
    auto lhs = extract_value(call_["args"][0]);
    auto rhs = extract_value(call_["args"][1]);

    auto bin_op = create_binary_op(function, lhs, rhs);

    exprt e = converter_.get_expr(bin_op);

    auto dtype_size(get_dtype_size());
    if (dtype_size)
    {
      typet t = get_typet_from_dtype();
      if (converter_.current_lhs)
      {
        converter_.current_lhs->type() = t;
        converter_.update_symbol(*converter_.current_lhs);

        e.type() = converter_.current_lhs->type();
        if (!e.operands().empty())
        {
          e.operands().at(0).type() = e.type();
          e.operands().at(1).type() = e.type();
        }

        std::string value_str = e.value().as_string();
        size_t value_size = count_effective_bits(value_str);

        if (value_size > dtype_size)
        {
          log_warning(
            "{}:{}: Integer overflow detected in {}() call. Consider using a "
            "larger integer type.",
            converter_.current_python_file,
            call_["end_lineno"].get<int>(),
            function_id_.get_function());
        }

        if (!e.value().empty())
        {
          auto length = value_str.length();
          e.value(value_str.substr(length - dtype_size));
          value_str = e.value().as_string();
          e.set("#cformat", std::to_string(std::stoll(value_str, nullptr, 2)));
        }
      }
    }

    return e;
  }

  throw std::runtime_error("Unsupported NumPy function call: " + function);
}
