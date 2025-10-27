#include <python-frontend/numpy_call_expr.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/json_utils.h>
#include <util/expr.h>
#include <util/expr_util.h>
#include <util/c_types.h>
#include <util/message.h>
#include <util/arith_tools.h>

#include <ostream>

const char *kConstant = "Constant";
const char *kName = "Name";

numpy_call_expr::numpy_call_expr(
  const symbol_id &function_id,
  const nlohmann::json &call,
  python_converter &converter)
  : function_call_expr(function_id, call, converter)
{
  converter_.build_static_lists = true;
}

numpy_call_expr::~numpy_call_expr()
{
  converter_.build_static_lists = false;
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
static auto create_binary_op(
  const std::string &op,
  const std::string &type,
  const T &lhs,
  const T &rhs)
{
  nlohmann::json left, right;

  if (type == kName)
  {
    left = {{"_type", type}, {"id", lhs}};
    right = {{"_type", type}, {"id", rhs}};
  }
  else
  {
    left = {{"_type", type}, {"value", lhs}};
    right = {{"_type", type}, {"value", rhs}};
  }

  nlohmann::json bin_op = {
    {"_type", "BinOp"},
    {"left", left},
    {"op", {{"_type", op}}},
    {"right", right}};

  return bin_op;
}

bool numpy_call_expr::is_math_function() const
{
  const std::string &function = function_id_.get_function();
  return function == "add" || function == "subtract" ||
         function == "multiply" ||
         (function == "divide" || function == "power" || function == "ceil" ||
          function == "floor" || function == "fabs" || function == "sin" ||
          function == "cos" || function == "exp" || function == "fmod" ||
          function == "sqrt" || function == "fmin") ||
         function == "fmax" || function == "trunc" || function == "round" ||
         function == "arccos" || function == "copysign" ||
         function == "arctan" || function == "dot" || function == "transpose" ||
         function == "det" || function == "matmul";
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

template <typename T>
T get_constant_value(const nlohmann::json &node)
{
  if (node["_type"] == "Constant")
  {
    return node["value"].get<T>();
  }
  else if (node["_type"] == "UnaryOp" && node["operand"]["_type"] == "Constant")
  {
    std::string op_type = node["op"]["_type"];
    T val = node["operand"]["value"].get<T>();

    if (op_type == "USub")
      return -val;
    else if (op_type == "UAdd")
      return val;
    else
    {
      log_error("get_constant_value: Unsupported unary operator '{}'", op_type);
      abort();
    }
  }
  else
  {
    log_error(
      "get_constant_value: Expected Constant or UnaryOp with Constant operand, "
      "got '{}'",
      node.dump());
    abort();
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
    else if (arg_type == "List")
    {
      const std::string &operation = function_id_.get_function();
      if (operation == "transpose")
      {
        code_function_callt call =
          to_code_function_call(to_code(function_call_expr::get()));
        typet t = call.arguments().at(0).type().subtype();
        converter_.current_lhs->type() = t;
        converter_.update_symbol(*converter_.current_lhs);
        call.arguments().push_back(address_of_exprt(*converter_.current_lhs));
        std::vector<int> shape = type_handler_.get_array_type_shape(t);
        call.arguments().push_back(from_integer(shape[0], int_type()));
        call.arguments().push_back(from_integer(shape[1], int_type()));
        return call;
      }
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
        if (func_name == "ceil")
          func_name = "__" + func_name + "_array";
        function_id_.set_function(func_name);

        code_function_callt call =
          to_code_function_call(to_code(function_call_expr::get()));
        typet t = type_handler_.get_list_type(arg);

        // In a call like result = np.ceil(v), the type of 'result' is only known after processing the argument 'v'.
        // At this point, we have the argument's type information, so we update the type of the LHS expression accordingly.

        if (t.subtype().is_array())
          converter_.current_lhs->type() = long_long_int_type();
        else
          converter_.current_lhs->type() = t;

        converter_.update_symbol(*converter_.current_lhs);

        // NumPy math functions on arrays are translated to C-style calls with the signature: func(input, output, size).
        // For example, result = np.ceil(v) becomes ceil_array(v, result, sizeof(v)).
        // The lines below add the output array and size arguments to the call.

        // Add output argument
        call.arguments().push_back(address_of_exprt(*converter_.current_lhs));

        // Add array size arguments
        if (t.subtype().is_array())
        {
          std::vector<int> shape = type_handler_.get_array_type_shape(t);
          call.arguments().push_back(from_integer(shape[0], int_type()));
          call.arguments().push_back(from_integer(shape[1], int_type()));
        }
        else
        {
          exprt array_size = from_integer(arg["elts"].size(), int_type());
          call.arguments().push_back(array_size);
        }

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

    if (
      (lhs["_type"] == "Constant" || lhs["_type"] == "UnaryOp") &&
      (rhs["_type"] == "Constant" || rhs["_type"] == "UnaryOp"))
    {
      bool lhs_is_float =
        (lhs["_type"] == "UnaryOp" ? lhs["operand"]["value"].is_number_float()
                                   : lhs["value"].is_number_float());
      bool rhs_is_float =
        (rhs["_type"] == "UnaryOp" ? rhs["operand"]["value"].is_number_float()
                                   : rhs["value"].is_number_float());

      if (lhs_is_float || rhs_is_float)
      {
        double lhs_val = get_constant_value<double>(lhs);
        double rhs_val = get_constant_value<double>(rhs);
        expr = create_binary_op(
          function_id_.get_function(), kConstant, lhs_val, rhs_val);
      }
      else
      {
        int lhs_val = get_constant_value<int>(lhs);
        int rhs_val = get_constant_value<int>(rhs);
        expr = create_binary_op(
          function_id_.get_function(), kConstant, lhs_val, rhs_val);
      }
    }
    else if (lhs["_type"] == "AnnAssign" && rhs["_type"] == "AnnAssign")
    {
      expr = create_binary_op(
        function_id_.get_function(),
        kName,
        lhs["target"]["id"],
        rhs["target"]["id"]);
    }
    else if (lhs["_type"] == "List" && rhs["_type"] == "List")
    {
      // Get the name of the function being called (e.g., "dot" or "matmul")
      const std::string &operation = function_id_.get_function();

      if (operation == "dot" || operation == "matmul")
      {
        // Determine dimensionality of both operands
        bool lhs_is_2d = type_handler_.is_2d_array(lhs);
        bool rhs_is_2d = type_handler_.is_2d_array(rhs);

        size_t m, n, n2, p;
        typet base_type;

        if (!lhs_is_2d && !rhs_is_2d)
        {
          // 1D × 1D case: vector dot product
          size_t lhs_len = lhs["elts"].size();
          size_t rhs_len = rhs["elts"].size();

          if (lhs_len != rhs_len)
          {
            throw std::runtime_error("Incompatible shapes for dot product");
          }

          // Get element type from first element
          const auto &elem = lhs["elts"][0]["value"];
          base_type = type_handler_.get_typet(elem);

          // For 1D dot product, treat as (1×n) × (n×1) = (1×1) scalar
          m = 1;
          n = lhs_len;
          n2 = rhs_len;
          p = 1;

          // Result is a scalar, not a matrix
          converter_.current_lhs->type() = base_type;
        }
        else if (!lhs_is_2d && rhs_is_2d)
        {
          // 1D × 2D case: (n,) × (n, p) -> (p,)
          size_t lhs_len = lhs["elts"].size();
          size_t rhs_rows = rhs["elts"].size();
          size_t rhs_cols = rhs["elts"][0]["elts"].size();

          if (lhs_len != rhs_rows)
          {
            throw std::runtime_error("Incompatible shapes for dot product");
          }

          const auto &elem = rhs["elts"][0]["elts"][0]["value"];
          base_type = type_handler_.get_typet(elem);

          m = 1;
          n = lhs_len;
          n2 = rhs_rows;
          p = rhs_cols;

          // Result is 1D array of length p
          typet result_type = type_handler_.build_array(base_type, p);
          converter_.current_lhs->type() = result_type;
        }
        else if (lhs_is_2d && !rhs_is_2d)
        {
          // 2D × 1D case: (m, n) × (n,) -> (m,)
          size_t lhs_rows = lhs["elts"].size();
          size_t lhs_cols = lhs["elts"][0]["elts"].size();
          size_t rhs_len = rhs["elts"].size();

          if (lhs_cols != rhs_len)
          {
            throw std::runtime_error("Incompatible shapes for dot product");
          }

          const auto &elem = lhs["elts"][0]["elts"][0]["value"];
          base_type = type_handler_.get_typet(elem);

          m = lhs_rows;
          n = lhs_cols;
          n2 = rhs_len;
          p = 1;

          // Result is 1D array of length m
          typet result_type = type_handler_.build_array(base_type, m);
          converter_.current_lhs->type() = result_type;
        }
        else
        {
          // 2D × 2D case: original matrix multiplication logic
          m = lhs["elts"].size();
          n = lhs["elts"][0]["elts"].size();
          n2 = rhs["elts"].size();
          p = rhs["elts"][0]["elts"].size();

          if (n != n2)
          {
            throw std::runtime_error("Incompatible shapes for dot product");
          }

          const auto &elem = lhs["elts"][0]["elts"][0]["value"];
          base_type = type_handler_.get_typet(elem);

          // Build a (m × p) matrix type: array of m rows, each of p elements
          typet row_type = type_handler_.build_array(base_type, p);
          typet matrix_type = type_handler_.build_array(row_type, m);
          converter_.current_lhs->type() = matrix_type;
        }

        // Normalize to "dot" regardless of whether "matmul" was originally used
        function_id_.set_function("dot");
        // Update the symbol associated with the result
        converter_.update_symbol(*converter_.current_lhs);

        // Generate a function call expression to the backend `dot` function
        code_function_callt call =
          to_code_function_call(to_code(function_call_expr::get()));

        // Arguments:
        // 1. Output pointer
        // 2. m = number of rows in lhs (or 1 for 1D lhs)
        // 3. n = inner dimension (shared)
        // 4. p = number of columns in rhs (or 1 for 1D rhs)
        auto &args = call.arguments();
        args.push_back(address_of_exprt(*converter_.current_lhs));
        args.push_back(from_integer(m, int_type()));
        args.push_back(from_integer(n, int_type()));
        args.push_back(from_integer(p, int_type()));

        return call;
      }
      // Handle other binary operations like add, subtract, multiply, divide
      if (
        operation == "add" || operation == "subtract" ||
        operation == "multiply" || operation == "divide")
      {
        code_function_callt call =
          to_code_function_call(to_code(function_call_expr::get()));
        typet size = type_handler_.get_typet(lhs["elts"]);
        typet t = converter_.get_static_array(lhs, size).type();

        converter_.current_lhs->type() = t;
        converter_.update_symbol(*converter_.current_lhs);
        auto &args = call.arguments();
        args.push_back(address_of_exprt(*converter_.current_lhs));

        std::vector<int> shape = type_handler_.get_array_type_shape(t);
        exprt m = shape.size() < 2 ? gen_one(int_type())
                                   : from_integer(shape[0], int_type());
        exprt n = from_integer(shape.back(), int_type());
        args.push_back(m);
        args.push_back(n);

        return call;
      }

      throw std::runtime_error("Unsupported operation: " + operation);
    }
  }

  if (expr.empty())
    throw std::runtime_error(
      "Unsupported Numpy call: " + function_id_.get_function());

  return converter_.get_expr(expr);
}

exprt numpy_call_expr::get()
{
  const std::string &function = function_id_.get_function();

  // Create array from numpy.array()
  if (function == "array")
  {
    // Check for 3D+ arrays and reject them early
    int array_dims = type_handler_.get_array_dimensions(call_["args"][0]);

    if (array_dims >= 3)
    {
      throw std::runtime_error(
        "ESBMC does not support 3D or higher dimensional arrays. "
        "Found " +
        std::to_string(array_dims) +
        "D array creation. "
        "Please use 1D or 2D arrays only.");
    }

    typet size = type_handler_.get_typet(call_["args"][0]["elts"]);
    return converter_.get_static_array(call_["args"][0], size);
  }

  static const std::unordered_map<std::string, float> array_creation_funcs = {
    {"zeros", 0.0}, {"ones", 1.0}};

  // Create array from numpy.zeros() or numpy.ones()
  auto it = array_creation_funcs.find(function);
  if (it != array_creation_funcs.end())
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

        // Update all operands' types safely
        for (auto &operand : expr.operands())
          operand.type() = expr.type();

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
