#include <python-frontend/function_call_expr.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/json_utils.h>
#include <util/c_typecast.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/string_constant.h>
#include <regex>

using namespace json_utils;

function_call_expr::function_call_expr(
  const symbol_id &function_id,
  const nlohmann::json &call,
  python_converter &converter)
  : function_id_(function_id),
    call_(call),
    converter_(converter),
    type_handler_(converter.get_type_handler()),
    function_type_(FunctionType::FreeFunction)
{
  get_function_type();
}

static std::string get_classname_from_symbol_id(const std::string &symbol_id)
{
  // This function might return "Base" for a symbol_id as: py:main.py@C@Base@F@foo@self

  std::string class_name;
  size_t class_pos = symbol_id.find("@C@");
  size_t func_pos = symbol_id.find("@F@");

  if (class_pos != std::string::npos && func_pos != std::string::npos)
  {
    size_t length = func_pos - (class_pos + 3); // "+3" to ignore "@C@"
    // Extract substring between "@C@" and "@F@"
    class_name = symbol_id.substr(class_pos + 3, length);
  }
  return class_name;
}

void function_call_expr::get_function_type()
{
  if (type_handler_.is_constructor_call(call_))
  {
    function_type_ = FunctionType::Constructor;
    return;
  }

  if (call_["func"]["_type"] == "Attribute")
  {
    std::string caller = get_object_name();

    // Handling a function call as a class method call when:
    // (1) The caller corresponds to a class name, for example: MyClass.foo().
    // (2) Calling methods of built-in types, such as int.from_bytes()
    //     All the calls to built-in methods are handled by class methods in operational models.
    // (3) Calling a instance method from a built-in type object, for example: x.bit_length() when x is an int
    // If the caller is a class or a built-in type, the following condition detects a class method call.
    if (
      is_class(caller, converter_.ast()) ||
      type_utils::is_builtin_type(caller) ||
      type_utils::is_builtin_type(type_handler_.get_var_type(caller)))
    {
      function_type_ = FunctionType::ClassMethod;
    }
    else if (!converter_.is_imported_module(caller))
    {
      function_type_ = FunctionType::InstanceMethod;
    }
  }
}

bool function_call_expr::is_nondet_call() const
{
  std::regex pattern(
    R"(nondet_(int|char|bool|float)|__VERIFIER_nondet_(int|char|bool|float))");

  return std::regex_match(function_id_.get_function(), pattern);
}

exprt function_call_expr::build_nondet_call() const
{
  const std::string &func_name = function_id_.get_function();

  // Function name pattern: nondet_(type). e.g: nondet_bool(), nondet_int()
  size_t underscore_pos = func_name.rfind("_");
  std::string type = func_name.substr(underscore_pos + 1);
  exprt rhs = exprt("sideeffect", type_handler_.get_typet(type));
  rhs.statement("nondet");
  return rhs;
}

exprt function_call_expr::handle_hex(nlohmann::json &arg) const
{
  long long int_value = 0;
  bool is_negative = false;

  if (arg.contains("_type") && arg["_type"] == "UnaryOp")
  {
    const auto &op = arg["op"];
    const auto &operand = arg["operand"];

    if (
      op["_type"] == "USub" && operand.contains("value") &&
      operand["value"].is_number_integer())
    {
      is_negative = true;
      int_value = operand["value"].get<long long>();
    }
    else
      throw std::runtime_error("TypeError: Unsupported UnaryOp in hex()");
  }
  else if (arg.contains("value") && arg["value"].is_number_integer())
  {
    int_value = arg["value"].get<long long>();
    if (int_value < 0)
      is_negative = true;
  }
  else
    throw std::runtime_error("TypeError: hex() argument must be an integer");

  std::ostringstream oss;
  oss << (is_negative ? "-0x" : "0x") << std::hex << std::nouppercase
      << std::llabs(int_value);
  const std::string hex_str = oss.str();

  typet t = type_handler_.get_typet("str", hex_str.size());
  std::vector<uint8_t> string_literal(hex_str.begin(), hex_str.end());
  return converter_.make_char_array_expr(string_literal, t);
}

exprt function_call_expr::build_constant_from_arg() const
{
  const std::string &func_name = function_id_.get_function();
  size_t arg_size = 1;
  auto arg = call_["args"][0];

  // Handle str(): determine size of the resulting string constant
  if (func_name == "str")
  {
    if (!arg.contains("value") || !arg["value"].is_string())
      throw std::runtime_error("TypeError: str() expects a string argument");

    const std::string &s = arg["value"].get<std::string>();
    arg_size = s.size();
  }

  // Handle int(): convert float to int
  else if (func_name == "int" && arg["value"].is_number_float())
  {
    double arg_value = arg["value"].get<double>();
    arg["value"] = static_cast<int>(arg_value);
  }

  // Handle float(): convert int to float
  else if (func_name == "float" && arg["value"].is_number_integer())
  {
    int arg_value = arg["value"].get<int>();
    arg["value"] = static_cast<double>(arg_value);
  }

  // Handle chr(): convert integer to single-character string
  else if (func_name == "chr")
  {
    int int_value = 0;

    // Check for unary minus: e.g., chr(-1)
    if (arg.contains("_type") && arg["_type"] == "UnaryOp")
    {
      const auto &op = arg["op"];
      const auto &operand = arg["operand"];

      if (
        op["_type"] == "USub" && operand.contains("value") &&
        operand["value"].is_number_integer())
        int_value = -operand["value"].get<int>();
      else
        throw std::runtime_error("TypeError: Unsupported UnaryOp in chr()");
    }

    // Handle integer input
    else if (arg.contains("value") && arg["value"].is_number_integer())
      int_value = arg["value"].get<int>();

    // Reject float input
    else if (arg.contains("value") && arg["value"].is_number_float())
      throw std::runtime_error(
        "TypeError: chr() argument must be int, not float");

    // Try converting string input to integer
    else if (arg.contains("value") && arg["value"].is_string())
    {
      const std::string &s = arg["value"].get<std::string>();
      try
      {
        int_value = std::stoi(s);
      }
      catch (const std::invalid_argument &)
      {
        throw std::runtime_error(
          "TypeError: invalid string passed to chr(): '" + s + "'");
      }
    }

    // Validate Unicode range: [0, 0x10FFFF]
    if (int_value < 0 || int_value > 0x10FFFF)
    {
      throw std::runtime_error(
        "ValueError: chr() argument out of valid Unicode range: " +
        std::to_string(int_value));
    }

    // Replace the value with a single-character string
    arg["value"] = std::string(1, static_cast<char>(int_value));
    arg_size = 1;
  }

  else if (func_name == "hex")
    return handle_hex(arg);

  // Construct expression with appropriate type
  typet t = type_handler_.get_typet(func_name, arg_size);
  exprt expr = converter_.get_expr(arg);
  expr.type() = t;

  return expr;
}

std::string function_call_expr::get_object_name() const
{
  const auto &subelement = call_["func"]["value"];

  std::string obj_name;
  if (subelement["_type"] == "Attribute")
    obj_name = subelement["attr"].get<std::string>();
  else if (subelement["_type"] == "Constant" || subelement["_type"] == "BinOp")
    obj_name = function_id_.get_class();
  else
    obj_name = subelement["id"].get<std::string>();

  return json_utils::get_object_alias(converter_.ast(), obj_name);
}

exprt function_call_expr::get()
{
  // Handle non-det functions
  if (is_nondet_call())
  {
    return build_nondet_call();
  }

  const std::string &func_name = function_id_.get_function();

  /* Calls to initialise variables using built-in type functions such as int(1), str("test"), bool(1)
   * are converted to simple variable assignments, simplifying the handling of built-in type objects.
   * For example, x = int(1) becomes x = 1. */
  if (
    type_utils::is_builtin_type(func_name) ||
    type_utils::is_consensus_type(func_name))
  {
    return build_constant_from_arg();
  }

  auto &symbol_table = converter_.symbol_table();

  // Get object symbol
  symbolt *obj_symbol = nullptr;
  symbol_id obj_symbol_id = converter_.create_symbol_id();

  if (call_["func"]["_type"] == "Attribute")
  {
    std::string caller = get_object_name();
    obj_symbol_id.set_object(caller);
    obj_symbol = symbol_table.find_symbol(obj_symbol_id.to_string());
  }

  // Get function symbol id
  const std::string &func_symbol_id = function_id_.to_string();
  assert(!func_symbol_id.empty());

  // Find function symbol
  const symbolt *func_symbol = converter_.find_symbol(func_symbol_id.c_str());

  if (func_symbol == nullptr)
  {
    if (
      function_type_ == FunctionType::Constructor ||
      function_type_ == FunctionType::InstanceMethod)
    {
      // Get method from a base class when it is not defined in the current class
      func_symbol = converter_.find_function_in_base_classes(
        function_id_.get_class(),
        func_symbol_id,
        func_name,
        function_type_ == FunctionType::Constructor);

      if (function_type_ == FunctionType::Constructor)
      {
        if (!func_symbol)
        {
          // If __init__() is not defined for the class and bases,
          // an assignment (x = MyClass()) is converted to a declaration (x:MyClass) in python_converter::get_var_assign().
          return exprt("_init_undefined");
        }
        converter_.base_ctor_called = true;
      }
      else if (function_type_ == FunctionType::InstanceMethod)
      {
        assert(obj_symbol);

        // Update obj attributes from self
        converter_.update_instance_from_self(
          get_classname_from_symbol_id(func_symbol->id.as_string()),
          func_name,
          obj_symbol_id.to_string());
      }
    }
    else
    {
      log_warning("Undefined function: {}", func_name.c_str());
      return exprt();
    }
  }

  locationt location = converter_.get_location_from_decl(call_);

  code_function_callt call;
  call.location() = location;
  call.function() = symbol_expr(*func_symbol);
  const typet &return_type = to_code_type(func_symbol->type).return_type();
  call.type() = return_type;

  // Add self as first parameter
  if (function_type_ == FunctionType::Constructor)
  {
    // Self is the LHS
    assert(converter_.current_lhs);
    call.arguments().push_back(gen_address_of(*converter_.current_lhs));
  }
  else if (function_type_ == FunctionType::InstanceMethod)
  {
    assert(obj_symbol);
    // Passing object as "self" (first) parameter on instance method calls
    call.arguments().push_back(gen_address_of(symbol_expr(*obj_symbol)));
  }
  else if (function_type_ == FunctionType::ClassMethod)
  {
    // Passing a void pointer to the "cls" argument
    typet t = pointer_typet(empty_typet());
    call.arguments().push_back(gen_zero(t));

    // All methods for the int class without parameters acts solely on the encapsulated integer value.
    // Therefore, we always pass the caller (obj) as a parameter in these functions.
    // For example, if x is an int instance, x.bit_length() call becomes bit_length(x)
    if (
      obj_symbol &&
      type_handler_.get_var_type(obj_symbol->name.as_string()) == "int" &&
      call_["args"].empty())
    {
      call.arguments().push_back(symbol_expr(*obj_symbol));
    }
    else if (call_["func"]["value"]["_type"] == "BinOp")
    {
      // Handling function call from binary expressions like: (x+1).bit_length()
      call.arguments().push_back(converter_.get_expr(call_["func"]["value"]));
    }
  }

  for (const auto &arg_node : call_["args"])
  {
    exprt arg = converter_.get_expr(arg_node);

    // All array function arguments (e.g. bytes type) are handled as pointers.
    if (arg.type().is_array())
    {
      if (arg_node["_type"] == "Constant" && arg_node["value"].is_string())
      {
        arg = string_constantt(
          arg_node["value"].get<std::string>(),
          arg.type(),
          string_constantt::k_default);
      }
      call.arguments().push_back(address_of_exprt(arg));
    }
    else
      call.arguments().push_back(arg);
  }

  return std::move(call);
}
