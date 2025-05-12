#include <python-frontend/numpy_call_expr.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/json_utils.h>
#include <util/expr.h>
#include <util/c_types.h>
#include <util/message.h>
#include <util/arith_tools.h>

#include <ostream>

numpy_call_expr::numpy_call_expr(
  const symbol_id &function_id,
  const nlohmann::json &call,
  python_converter &converter)
  : function_call_expr(function_id, call, converter)
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
static auto create_list(const std::vector<T> &vector)
{
  nlohmann::json list;
  list["_type"] = "List";
  for (const auto &v : vector)
  {
    list["elts"].push_back({{"_type", "Constant"}, {"value", v}});
  }
  return list;
}

template <typename T>
static auto create_binary_op(const std::string &op, const T &lhs, const T &rhs)
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
         (function == "power") || (function == "ceil") ||
         (function == "floor") || (function == "fabs") || (function == "sin") ||
         (function == "cos") || (function == "exp") || (function == "fmod") ||
         (function == "sqrt") || (function == "fmin") || (function == "fmax") ||
         (function == "trunc");
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

// Checks if two shapes are broadcast-compatible.
// Two dimensions are compatible if they are equal or if one of them is 1.
bool is_broadcastable(
  const std::vector<int> &shape1,
  const std::vector<int> &shape2)
{
  int s1 = shape1.size() - 1;
  int s2 = shape2.size() - 1;

  // Compare dimensions from rightmost (inner) to leftmost (outer)
  while (s1 >= 0 || s2 >= 0)
  {
    // If a shape lacks a dimension, assume its size is 1.
    int d1 = (s1 >= 0) ? shape1[s1] : 1;
    int d2 = (s2 >= 0) ? shape2[s2] : 1;

    // Check if dimensions are compatible (either equal or one is 1)
    if (d1 != d2 && d1 != 1 && d2 != 1)
      return false;

    --s1;
    --s2;
  }
  return true;
}

void numpy_call_expr::broadcast_check(const nlohmann::json &operands) const
{
  std::vector<int> previous_shape;
  bool is_first_operand = true;
  symbol_id sid = converter_.create_symbol_id();

  for (const auto &op : operands)
  {
    if (op["_type"] == "Name")
    {
      sid.set_object(op["id"].get<std::string>());
      symbolt *s = converter_.find_symbol(sid.to_string());
      assert(s);

      // Retrieve the current operand's array shape.
      std::vector<int> current_shape =
        converter_.type_handler_.get_array_type_shape(s->type);

      // For subsequent operands, compare shapes using broadcasting rules.
      if (!is_first_operand)
      {
        if (!is_broadcastable(previous_shape, current_shape))
        {
          std::ostringstream oss;
          oss << "operands could not be broadcast together with shapes (";
          oss << previous_shape[0] << ",) (";
          oss << current_shape[0] << ",)";
          throw std::runtime_error(oss.str());
        }
      }
      else
      {
        is_first_operand = false;
      }

      // Update previous_shape for the next iteration.
      previous_shape = current_shape;
    }
  }
}

exprt numpy_call_expr::create_expr_from_call()
{
  nlohmann::json expr;

  // Resolve variables if they are names
  auto resolve_var = [this](nlohmann::json &var) {
    if (var["_type"] == "Name")
    {
      var = json_utils::find_var_decl(
        var["id"], function_id_.get_function(), converter_.ast());
      if (var["value"]["_type"] == "Call")
        var = var["value"]["args"][0];
    }
  };

  // Unary operations
  if (call_["args"].size() == 1)
  {
    const auto &arg_type = call_["args"][0]["_type"];
    if (
      arg_type == "Constant" || arg_type == "UnaryOp" ||
      arg_type == "Subscript")
    {
      return function_call_expr::get();
    }
    else if (arg_type == "Name")
    {
      auto arg = call_["args"][0];
      resolve_var(arg);

      // Handle calls with arrays as parameters; e.g. np.ceil([1, 2, 3])
      if (arg["_type"] == "List")
      {
        // Append array postfix to call array variants, e.g., ceil_array instead of ceil
        std::string func_name = function_id_.get_function();
        func_name = "__" + func_name + "_array";
        function_id_.set_function(func_name);

        code_function_callt call =
          to_code_function_call(to_code(function_call_expr::get()));
        typet t = type_handler_.get_list_type(arg);

        // In a call like result = np.ceil(v), the type of 'result' is only known after processing the argument 'v'.
        // At this point, we have the argument's type information, so we update the type of the LHS expression accordingly.

        converter_.current_lhs->type() = t;
        converter_.update_symbol(*converter_.current_lhs);

        // NumPy math functions on arrays are translated to C-style calls with the signature: func(input, output, size).
        // For example, result = np.ceil(v) becomes ceil_array(v, result, sizeof(v)).
        // The lines below add the output array and size arguments to the call.

        call.arguments().push_back(address_of_exprt(*converter_.current_lhs));
        exprt array_size = from_integer(arg["elts"].size(), int_type());
        call.arguments().push_back(array_size);

        return call;
      }
    }
  }

  // Binary operations
  if (call_["args"].size() == 2)
  {
    auto lhs = call_["args"][0];
    auto rhs = call_["args"][1];

    resolve_var(lhs);
    resolve_var(rhs);

    if (lhs["_type"] == "Constant" && rhs["_type"] == "Constant")
    {
      expr = create_binary_op(
        function_id_.get_function(), lhs["value"], rhs["value"]);
    }
    else if (lhs["_type"] == "List" && rhs["_type"] == "List")
    {
      std::vector<int> res;
      const std::string &operation = function_id_.get_function();
      for (size_t i = 0; i < lhs["elts"].size(); ++i)
      {
        int left_val = lhs["elts"][i]["value"].get<int>();
        int right_val = rhs["elts"][i]["value"].get<int>();

        if (operation == "add")
          res.push_back(left_val + right_val);
        else if (operation == "subtract")
          res.push_back(left_val - right_val);
        else if (operation == "multiply")
          res.push_back(left_val * right_val);
        else if (operation == "divide")
        {
          if (right_val == 0)
            throw std::runtime_error("Division by zero in list operation");
          res.push_back(left_val / right_val);
        }
        else
        {
          throw std::runtime_error("Unsupported operation: " + operation);
        }
      }
      expr = create_list(res);
    }
  }

  if (expr.empty())
  {
    std::ostringstream oss;
    oss << "Unsupported NumPy call: " << function_id_.get_function();
    log_error("{}", oss.str());
    abort();
  }

  return converter_.get_expr(expr);
}

exprt numpy_call_expr::get()
{
  static const std::unordered_map<std::string, float> numpy_functions = {
    {"zeros", 0.0}, {"ones", 1.0}};

  const std::string &function = function_id_.get_function();

  // Create array from numpy.array()
  if (function == "array")
  {
    auto expr = converter_.get_expr(call_["args"][0]);
    return expr;
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
    broadcast_check(call_["args"]);

    exprt expr = create_expr_from_call();

    auto dtype_size(get_dtype_size());
    if (dtype_size)
    {
      typet t = get_typet_from_dtype();
      if (converter_.current_lhs)
      {
        // Update variable (lhs)
        converter_.current_lhs->type() = t;
        converter_.update_symbol(*converter_.current_lhs);

        // Update rhs expression
        expr.type() = converter_.current_lhs->type();
        if (!expr.operands().empty())
        {
          expr.operands().at(0).type() = expr.type();
          expr.operands().at(1).type() = expr.type();
        }

        std::string value_str = expr.value().as_string();
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

        if (!expr.value().empty())
        {
          auto length = value_str.length();
          expr.value(value_str.substr(length - dtype_size));
          value_str = expr.value().as_string();
          expr.set(
            "#cformat", std::to_string(std::stoll(value_str, nullptr, 2)));
        }
      }
    }

    return expr;
  }

  throw std::runtime_error("Unsupported NumPy function call: " + function);
}
