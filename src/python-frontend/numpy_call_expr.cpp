#include <python-frontend/numpy_call_expr.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/python_converter.h>
#include <util/expr.h>
#include <util/c_types.h>
#include <util/message.h>
#include <variant>

using value_type = std::variant<int64_t, double>;

numpy_call_expr::numpy_call_expr(
  const symbol_id &function_id,
  const nlohmann::json &call,
  python_converter &converter)
  : function_id_(function_id), call_(call), converter_(converter)
{
}

// Extracts numerical values ​​from JSON, supporting int and double
static value_type extract_value(const nlohmann::json &arg)
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
    auto operand = arg["operand"]["value"];

    if (operand.is_number_integer())
    {
      return -operand.get<int64_t>();
    }
    else if (operand.is_number_float())
    {
      return -operand.get<double>();
    }
  }

  if (!arg.contains("value"))
  {
    throw std::runtime_error("Invalid JSON: missing value");
  }

  auto value = arg["value"];

  if (value.is_number_integer())
  {
    return value.get<int64_t>();
  }
  else if (value.is_number_float())
  {
    return value.get<double>();
  }

  throw std::runtime_error("Unknown numeric type in JSON");
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

    // Performs binary operation with support for int and double
    auto result = std::visit(
        [&](auto l, auto r) -> nlohmann::json {
            using LType = decltype(l);
            using RType = decltype(r);

            if constexpr (std::is_same_v<LType, int64_t> && std::is_same_v<RType, int64_t>)
            {
                return create_binary_op(function, l, r);
            }

            double left = std::holds_alternative<int64_t>(lhs) ? static_cast<double>(std::get<int64_t>(lhs)) : std::get<double>(lhs);
            double right = std::holds_alternative<int64_t>(rhs) ? static_cast<double>(std::get<int64_t>(rhs)) : std::get<double>(rhs);

            return create_binary_op(function, left, right);
        },
        lhs, rhs);

    exprt e = converter_.get_expr(result);

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

        std::string dtype = get_dtype();
        auto final_value = std::visit([](auto v) -> double { return v; }, lhs) + std::visit([](auto v) -> double { return v; }, rhs);

        if (dtype.find("int") != std::string::npos)
        {
          int64_t mask = (1LL << dtype_size) - 1;
          final_value = static_cast<int64_t>(final_value) & mask;

          if (dtype[0] != 'u' && (static_cast<int64_t>(final_value) >> (dtype_size - 1)) & 1)
          {
            final_value -= (1LL << dtype_size);
          }
        
        log_warning(
          "{}:{}: Integer overflow detected in {}() call. Consider using a "
          "larger integer type.",
          converter_.current_python_file,
          call_["end_lineno"].get<int>(),
          function_id_.get_function());
        }

        e.set("#cformat", std::to_string(final_value));
      }
    }

    return e;
  }

  throw std::runtime_error("Unsupported NumPy function call: " + function);
}
