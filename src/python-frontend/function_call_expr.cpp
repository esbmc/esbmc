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
#include <util/arith_tools.h>
#include <util/message.h>

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

exprt function_call_expr::handle_int_to_str(nlohmann::json &arg) const
{
  std::string str_val = std::to_string(arg["value"].get<int>());
  // Convert string to vector of unsigned char
  std::vector<unsigned char> chars(str_val.begin(), str_val.end());
  // Get type for the array
  typet t = type_handler_.get_typet("str", chars.size());
  // Use helper to generate constant string expression
  return converter_.make_char_array_expr(chars, t);
}

exprt function_call_expr::handle_float_to_str(nlohmann::json &arg) const
{
  std::string str_val = std::to_string(arg["value"].get<double>());

  // Remove unnecessary trailing zeros and dot if needed (to match Python str behavior)
  // Example: "5.500000" → "5.5"
  str_val.erase(str_val.find_last_not_of('0') + 1, std::string::npos);
  if (str_val.back() == '.')
    str_val.pop_back();

  std::vector<unsigned char> chars(str_val.begin(), str_val.end());
  typet t = type_handler_.get_typet("str", chars.size());
  return converter_.make_char_array_expr(chars, t);
}

size_t function_call_expr::handle_str(nlohmann::json &arg) const
{
  if (!arg.contains("value") || !arg["value"].is_string())
    throw std::runtime_error("TypeError: str() expects a string argument");

  return arg["value"].get<std::string>().size();
}

void function_call_expr::handle_float_to_int(nlohmann::json &arg) const
{
  double value = arg["value"].get<double>();
  arg["value"] = static_cast<int>(value);
}

void function_call_expr::handle_int_to_float(nlohmann::json &arg) const
{
  int value = arg["value"].get<int>();
  arg["value"] = static_cast<double>(value);
}

void function_call_expr::handle_chr(nlohmann::json &arg) const
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

exprt function_call_expr::handle_oct(nlohmann::json &arg) const
{
  long long int_value = 0;  // Holds the integer value to be converted
  bool is_negative = false; // Tracks if the number is negative

  // Check if the argument is a unary operation (like -123)
  if (arg.contains("_type") && arg["_type"] == "UnaryOp")
  {
    const auto &op = arg["op"];           // Operator (e.g., USub)
    const auto &operand = arg["operand"]; // Operand of the unary operator

    // Only support unary subtraction (-) of integer literals
    if (
      op["_type"] == "USub" && operand.contains("value") &&
      operand["value"].is_number_integer())
    {
      int_value = operand["value"].get<long long>();

      // Treat -0 as 0 (consistent with Python behavior)
      if (int_value != 0)
        is_negative = true;
    }
    else
      throw std::runtime_error("TypeError: Unsupported UnaryOp in oct()");
  }
  // If it's not a unary operation, expect a plain integer literal
  else if (arg.contains("value") && arg["value"].is_number_integer())
  {
    int_value = arg["value"].get<long long>();
    if (int_value < 0)
      is_negative = true;
  }
  else
  {
    // Invalid argument type for oct()
    throw std::runtime_error("TypeError: oct() argument must be an integer");
  }

  // Convert the absolute value to octal and prepend "0o" (or "-0o")
  std::ostringstream oss;
  oss << (is_negative ? "-0o" : "0o") << std::oct << std::llabs(int_value);
  const std::string oct_str = oss.str();

  // Create a string type and return a character array expression
  typet t = type_handler_.get_typet("str", oct_str.size());
  std::vector<uint8_t> string_literal(oct_str.begin(), oct_str.end());
  return converter_.make_char_array_expr(string_literal, t);
}

exprt function_call_expr::handle_ord(nlohmann::json &arg) const
{
  int code_point = 0;

  // Ensure the argument is a string
  if (arg.contains("value") && arg["value"].is_string())
  {
    const std::string &s = arg["value"].get<std::string>();
    const unsigned char *bytes =
      reinterpret_cast<const unsigned char *>(s.c_str());
    size_t length = s.length();

    if (length == 0)
    {
      throw std::runtime_error(
        "TypeError: ord() expected a character, but string of length 0 found");
    }

    // Manual UTF-8 decoding
    if ((bytes[0] & 0x80) == 0)
    {
      // 1-byte ASCII
      if (length != 1)
        throw std::runtime_error(
          "TypeError: ord() expected a single character");

      code_point = bytes[0];
    }
    else if ((bytes[0] & 0xE0) == 0xC0)
    {
      // 2-byte sequence
      if (length != 2)
        throw std::runtime_error(
          "TypeError: ord() expected a single character");

      code_point = ((bytes[0] & 0x1F) << 6) | (bytes[1] & 0x3F);
    }
    else if ((bytes[0] & 0xF0) == 0xE0)
    {
      // 3-byte sequence
      if (length != 3)
        throw std::runtime_error(
          "TypeError: ord() expected a single character");

      code_point = ((bytes[0] & 0x0F) << 12) | ((bytes[1] & 0x3F) << 6) |
                   (bytes[2] & 0x3F);
    }
    else if ((bytes[0] & 0xF8) == 0xF0)
    {
      // 4-byte sequence
      if (length != 4)
        throw std::runtime_error(
          "TypeError: ord() expected a single character");

      code_point = ((bytes[0] & 0x07) << 18) | ((bytes[1] & 0x3F) << 12) |
                   ((bytes[2] & 0x3F) << 6) | (bytes[3] & 0x3F);
    }
    else
    {
      throw std::runtime_error(
        "ValueError: ord() received invalid UTF-8 input");
    }
  }
  else
  {
    throw std::runtime_error("TypeError: ord() argument must be a string");
  }

  // Replace the arg with the integer value
  arg["value"] = code_point;
  arg["type"] = "int";

  // Build and return the integer expression
  exprt expr = converter_.get_expr(arg);
  expr.type() = type_handler_.get_typet("int", 0);
  return expr;
}

exprt function_call_expr::handle_str_symbol_to_int(const symbolt *sym) const
{
  std::string value;
  const exprt &val = sym->value;

  // Case 1: value is an array of characters (i.e., a proper string)
  if(val.type().is_array() && val.has_operands())
  {
    for(const auto &ch : val.operands())
    {
      try
      {
        const auto &const_expr = to_constant_expr(ch);
        std::string binary_str = id2string(const_expr.get_value());

        // Interpret value as binary string
        unsigned c = std::stoul(binary_str, nullptr, 2);

        if(c == 0) break; // null terminator
        if(c < 32 || c > 126)
        {
          log_error("Invalid character code in string (non-printable): {}", c);
          return from_integer(0, type_handler_.get_typet("int", 0));
        }

        value += static_cast<char>(c);
      }
      catch(const std::exception &e)
      {
        log_error("Exception during character extraction: {}", e.what());
        return from_integer(0, type_handler_.get_typet("int", 0));
      }
    }
  }
  // Case 2: value is a single character constant
  else if(val.is_constant() && val.type().is_signedbv())
  {
    try
    {
      std::string binary_str = id2string(to_constant_expr(val).get_value());
      unsigned c = std::stoul(binary_str, nullptr, 2);

      if(c < 32 || c > 126)
      {
        log_error("Invalid character code (non-printable): {}", c);
        return from_integer(0, type_handler_.get_typet("int", 0));
      }

      value += static_cast<char>(c);
    }
    catch(const std::exception &e)
    {
      log_error("Exception during single-char extraction: {}", e.what());
      return from_integer(0, type_handler_.get_typet("int", 0));
    }
  }
  else
  {
    log_error("Unhandled symbol format for int() conversion.");
    return from_integer(0, type_handler_.get_typet("int", 0));
  }

  log_status("Reconstructed string value: \"{}\"", value);

  // Validate that it's a digit-only string
  if(value.empty() || !std::all_of(value.begin(), value.end(), ::isdigit))
  {
    log_error("Invalid string for integer conversion: \"{}\"", value);
    return from_integer(0, type_handler_.get_typet("int", 0));
  }

  int int_val = std::stoi(value);
  return from_integer(int_val, type_handler_.get_typet("int", 0));
}

exprt function_call_expr::build_constant_from_arg() const
{
  const std::string &func_name = function_id_.get_function();
  size_t arg_size = 1;
  auto arg = call_["args"][0];

  // Handle str(): convert int to str
  if (func_name == "str" && arg["value"].is_number_integer())
    return handle_int_to_str(arg);

  // Handle str(): convert float to str
  else if (func_name == "str" && arg["value"].is_number_float())
    return handle_float_to_str(arg);

  // Handle str(): determine size of the resulting string constant
  else if (func_name == "str")
    arg_size = handle_str(arg);

  // Handle int(): convert string (from symbol) to int
  else if (func_name == "int" && arg["_type"] == "Name")
  {
    std::string var_name = arg["id"];
    std::string filename = function_id_.get_filename();
    std::string var_symbol = "py:" + filename + "@" + var_name;

    // Look up the symbol using the variable name
    const symbolt *sym = converter_.find_symbol(var_symbol);

    if(!sym)
    {
      log_warning("Warning: symbol not found: {}", var_name);
      return from_integer(0, type_handler_.get_typet("int", 0));
    }

    if(sym->value.is_constant())
      return handle_str_symbol_to_int(sym);
  }

  // Handle int(): convert float to int
  else if (func_name == "int" && arg["value"].is_number_float())
    handle_float_to_int(arg);

  // Handle float(): convert int to float
  else if (func_name == "float" && arg["value"].is_number_integer())
    handle_int_to_float(arg);

  // Handle chr(): convert integer to single-character string
  else if (func_name == "chr")
    handle_chr(arg);

  // Handle ord(): convert single-character string to integer Unicode code point
  else if (func_name == "ord")
    return handle_ord(arg);

  // Handle hex: Handles hexadecimal string arguments
  else if (func_name == "hex")
    return handle_hex(arg);

  // Handle oct: Handles octal string arguments
  else if (func_name == "oct")
    return handle_oct(arg);

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
