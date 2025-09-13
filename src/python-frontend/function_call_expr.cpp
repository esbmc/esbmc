#include <python-frontend/function_call_expr.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/json_utils.h>
#include <util/base_type.h>
#include <util/c_typecast.h>
#include <util/expr_util.h>
#include <util/string_constant.h>
#include <util/arith_tools.h>
#include <util/ieee_float.h>
#include <util/message.h>
#include <regex>
#include <stdexcept>

using namespace json_utils;
namespace
{
// Constants for input handling
constexpr size_t MAX_INPUT_LENGTH = 256;

// Constants for UTF-8 encoding
constexpr unsigned int UTF8_1_BYTE_MAX = 0x7F;
constexpr unsigned int UTF8_2_BYTE_MAX = 0x7FF;
constexpr unsigned int UTF8_3_BYTE_MAX = 0xFFFF;
constexpr unsigned int UTF8_4_BYTE_MAX = 0x10FFFF;
constexpr unsigned int SURROGATE_START = 0xD800;
constexpr unsigned int SURROGATE_END = 0xDFFF;

// Constants for symbol parsing
constexpr const char *CLASS_MARKER = "@C@";
constexpr const char *FUNCTION_MARKER = "@F@";
} // namespace

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
  size_t class_pos = symbol_id.find(CLASS_MARKER);
  size_t func_pos = symbol_id.find(FUNCTION_MARKER);

  if (
    class_pos != std::string::npos && func_pos != std::string::npos &&
    func_pos > class_pos)
  {
    size_t start = class_pos + strlen(CLASS_MARKER);
    size_t length = func_pos - start;

    if (length > 0)
      class_name = symbol_id.substr(start, length);
  }
  return class_name;
}

int function_call_expr::decode_utf8_codepoint(const std::string &utf8_str) const
{
  if (utf8_str.empty())
  {
    throw std::runtime_error(
      "TypeError: expected a character, but string of length 0 found");
  }

  const unsigned char *bytes =
    reinterpret_cast<const unsigned char *>(utf8_str.c_str());
  size_t length = utf8_str.length();

  // Manual UTF-8 decoding
  if ((bytes[0] & 0x80) == 0)
  {
    // 1-byte ASCII
    if (length != 1)
      throw std::runtime_error("TypeError: expected a single character");
    return bytes[0];
  }
  else if ((bytes[0] & 0xE0) == 0xC0)
  {
    // 2-byte sequence
    if (length != 2)
      throw std::runtime_error("TypeError: expected a single character");
    return ((bytes[0] & 0x1F) << 6) | (bytes[1] & 0x3F);
  }
  else if ((bytes[0] & 0xF0) == 0xE0)
  {
    // 3-byte sequence
    if (length != 3)
      throw std::runtime_error("TypeError: expected a single character");
    return ((bytes[0] & 0x0F) << 12) | ((bytes[1] & 0x3F) << 6) |
           (bytes[2] & 0x3F);
  }
  else if ((bytes[0] & 0xF8) == 0xF0)
  {
    // 4-byte sequence
    if (length != 4)
      throw std::runtime_error("TypeError: expected a single character");
    return ((bytes[0] & 0x07) << 18) | ((bytes[1] & 0x3F) << 12) |
           ((bytes[2] & 0x3F) << 6) | (bytes[3] & 0x3F);
  }
  else
    throw std::runtime_error("ValueError: received invalid UTF-8 input");
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
  static std::regex pattern(
    R"(nondet_(int|char|bool|float)|__VERIFIER_nondet_(int|char|bool|float))");

  return std::regex_match(function_id_.get_function(), pattern);
}

bool function_call_expr::is_introspection_call() const
{
  const std::string &func_name = function_id_.get_function();
  return func_name == "isinstance" || func_name == "hasattr";
}

bool function_call_expr::is_input_call() const
{
  const std::string &func_name = function_id_.get_function();
  return func_name == "input";
}

exprt function_call_expr::handle_input() const
{
  // input() returns a non-deterministic string
  // We'll model input() as returning a non-deterministic string
  // with a reasonable maximum length (e.g., 256 characters)
  // This is an under-approximation to model the input function

  typet string_type = type_handler_.get_typet("str", MAX_INPUT_LENGTH);
  exprt rhs = exprt("sideeffect", string_type);
  rhs.statement("nondet");

  return rhs;
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

exprt function_call_expr::handle_isinstance() const
{
  const auto &args = call_["args"];

  // Ensure isinstance() is called with exactly two arguments
  if (args.size() != 2)
    throw std::runtime_error("isinstance() expects 2 arguments");

  // Convert the first argument (the object being checked) into an expression
  exprt obj_expr = converter_.get_expr(args[0]);

  // The second argument must be a simple type name (e.g., int, MyClass)
  if (args[1]["_type"] != "Name")
    throw std::runtime_error("Unsupported type format in isinstance()");

  std::string type_name = args[1]["id"];

  // Get the internal type representation from the type name
  typet expected_type = type_handler_.get_typet(type_name, 0);

  /* NOTE: Comparing the types directly may be insufficient.
           Inheritance or type aliases may require deeper analysis. */

  bool matches = base_type_eq(obj_expr.type(), expected_type, converter_.ns);
  if (matches)
    return gen_boolean(true);

  bool is_subtype =
    is_subclass_of(obj_expr.type(), expected_type, converter_.ns);
  return gen_boolean(is_subtype);
}

exprt function_call_expr::handle_hasattr() const
{
  const auto &args = call_["args"];
  symbol_id sid = converter_.create_symbol_id();
  sid.set_object(args[0]["id"]);
  bool has_attr = converter_.is_instance_attribute(
    sid.to_string(), args[1]["value"], sid.get_object(), "");
  return gen_boolean(has_attr);
}

exprt function_call_expr::handle_int_to_str(nlohmann::json &arg) const
{
  std::string str_val = std::to_string(arg["value"].get<int>());
  // Convert string to vector of unsigned char
  std::vector<unsigned char> chars(str_val.begin(), str_val.end());
  // Get type for the array
  typet t = type_handler_.get_typet("str", chars.size() + 1);
  // Use helper to generate constant string expression
  exprt str = converter_.make_char_array_expr(chars, t);
  return str;
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
  typet t = type_handler_.get_typet("str", chars.size() + 1);
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

std::string utf8_encode(unsigned int int_value)
{
  /**
   * Convert an integer value into its UTF-8 character equivalent
   * similar to the python chr() function
   */

  // Check for surrogate pairs (invalid in UTF-8)
  if (int_value >= SURROGATE_START && int_value <= SURROGATE_END)
  {
    std::ostringstream oss;
    oss << "Code point 0x" << std::hex << std::uppercase << int_value
        << " is a surrogate pair, invalid in UTF-8";
    throw std::invalid_argument(oss.str());
  }

  // Manual UTF-8 encoding
  std::string char_out;

  if (int_value <= UTF8_1_BYTE_MAX)
    char_out.append(1, static_cast<char>(int_value));
  else if (int_value <= UTF8_2_BYTE_MAX)
  {
    char_out.append(1, static_cast<char>(0xc0 | ((int_value >> 6) & 0x1f)));
    char_out.append(1, static_cast<char>(0x80 | (int_value & 0x3f)));
  }
  else if (int_value <= UTF8_3_BYTE_MAX)
  {
    char_out.append(1, static_cast<char>(0xe0 | ((int_value >> 12) & 0x0f)));
    char_out.append(1, static_cast<char>(0x80 | ((int_value >> 6) & 0x3f)));
    char_out.append(1, static_cast<char>(0x80 | (int_value & 0x3f)));
  }
  else if (int_value <= UTF8_4_BYTE_MAX)
  {
    char_out.append(1, static_cast<char>(0xf0 | ((int_value >> 18) & 0x07)));
    char_out.append(1, static_cast<char>(0x80 | ((int_value >> 12) & 0x3f)));
    char_out.append(1, static_cast<char>(0x80 | ((int_value >> 6) & 0x3f)));
    char_out.append(1, static_cast<char>(0x80 | (int_value & 0x3f)));
  }
  else
  {
    std::ostringstream oss;
    oss << "argument '0x" << std::hex << std::uppercase << int_value
        << "' outside of Unicode range: [0x000000, 0x110000)";
    throw std::out_of_range(oss.str());
  }
  return char_out;
}

exprt function_call_expr::handle_chr(nlohmann::json &arg) const
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
      return gen_exception_raise("TypeError", "Unsupported UnaryOp in chr()");
  }

  // Handle integer input
  else if (arg.contains("value") && arg["value"].is_number_integer())
    int_value = arg["value"].get<int>();

  // Reject float input
  else if (arg.contains("value") && arg["value"].is_number_float())
    return gen_exception_raise(
      "TypeError", "chr() argument must be int, not float");

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
      return gen_exception_raise("TypeError", "invalid string passed to chr()");
    }
  }

  else if (arg.contains("_type") && arg["_type"] == "Name")
  {
    const symbolt *sym = lookup_python_symbol(arg["id"]);
    if (!sym)
    {
      // if symbol not found revert to a variable assignment
      arg["value"] = std::string(1, static_cast<char>(int_value));
      typet t = type_handler_.get_typet("chr", 1);
      exprt expr = converter_.get_expr(arg);
      expr.type() = t;
      return expr;
    }
    exprt val = sym->value;

    if (!val.is_constant())
      val = converter_.get_resolved_value(val);

    if (val.is_nil())
      throw std::runtime_error(
        "Unable to resolve symbol " + arg["id"].get<std::string>());

    const auto &const_expr = to_constant_expr(val);
    std::string binary_str = id2string(const_expr.get_value());
    try
    {
      int_value = std::stoul(binary_str, nullptr, 2);
    }
    catch (std::out_of_range &)
    {
      return gen_exception_raise(
        "ValueError", "chr() argument outside of Unicode range");
    }
    catch (std::invalid_argument &)
    {
      return gen_exception_raise("TypeError", "must be of type int");
    }

    arg["_type"] = "Constant";
    arg.erase("id");
    arg.erase("ctx");
  }
  std::string utf8_encoded;

  try
  {
    utf8_encoded = utf8_encode(int_value);
  }
  catch (const std::out_of_range &e)
  {
    return gen_exception_raise("ValueError", "chr()");
  }

  // Replace the value with a single-character string
  arg["value"] = arg["n"] = arg["s"] = utf8_encoded;
  arg["type"] = "str";

  bool null_terminated = int_value > 0x7f;
  // Build and return the string expression
  exprt expr = converter_.get_expr(arg);
  expr.type() =
    type_handler_.get_typet("str", utf8_encoded.size() + null_terminated);
  return expr;
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
      return gen_exception_raise("TypeError", "Unsupported UnaryOp in hex()");
  }
  else if (arg.contains("value") && arg["value"].is_number_integer())
  {
    int_value = arg["value"].get<long long>();
    if (int_value < 0)
      is_negative = true;
  }
  else
    return gen_exception_raise(
      "TypeError", "hex() argument must be an integer");

  std::ostringstream oss;
  oss << (is_negative ? "-0x" : "0x") << std::hex << std::nouppercase
      << std::llabs(int_value);
  const std::string hex_str = oss.str();

  typet t = type_handler_.get_typet("str", hex_str.size() + 1);
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
      return gen_exception_raise("TypeError", "Unsupported UnaryOp in oct()");
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
    return gen_exception_raise(
      "TypeError", "oct() argument must be an integer");
  }

  // Convert the absolute value to octal and prepend "0o" (or "-0o")
  std::ostringstream oss;
  oss << (is_negative ? "-0o" : "0o") << std::oct << std::llabs(int_value);
  const std::string oct_str = oss.str();

  // Create a string type and return a character array expression
  typet t = type_handler_.get_typet("str", oct_str.size() + 1);
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
    code_point = decode_utf8_codepoint(s);
  }
  // Handle ord with symbol
  else if (arg["_type"] == "Name" && arg.contains("id"))
  {
    const symbolt *sym = lookup_python_symbol(arg["id"]);
    if (!sym)
    {
      std::string var_name = arg["id"].get<std::string>();
      return gen_exception_raise(
        "NameError", "variable '" + var_name + "' is not defined");
    }

    typet operand_type = sym->value.type();
    std::string py_type = type_handler_.type_to_string(operand_type);

    if (operand_type != char_type() && py_type != "str")
    {
      return gen_exception_raise(
        "TypeError",
        "ord() expected string of length 1, but " + py_type + " found");
    }

    auto value_opt = extract_string_from_symbol(sym);
    if (!value_opt)
    {
      return gen_exception_raise(
        "ValueError", "failed to extract string from symbol");
    }

    code_point = decode_utf8_codepoint(*value_opt);

    // Remove Name data
    arg["_type"] = "Constant";
    arg.erase("id");
    arg.erase("ctx");
  }
  else
    return gen_exception_raise("TypeError", "ord() argument must be a string");

  // Replace the arg with the integer value
  arg["value"] = code_point;
  arg["type"] = "int";

  // Build and return the integer expression
  exprt expr = converter_.get_expr(arg);
  expr.type() = type_handler_.get_typet("int", 0);
  return expr;
}

/// Extracts the character string represented by a symbol's constant value.
std::optional<std::string>
function_call_expr::extract_string_from_symbol(const symbolt *sym) const
{
  const exprt &val = sym->value;
  std::string result;

  auto decode_char = [](const exprt &expr) -> std::optional<char> {
    try
    {
      const auto &const_expr = to_constant_expr(expr);
      std::string binary_str = id2string(const_expr.get_value());
      unsigned c = std::stoul(binary_str, nullptr, 2);
      return static_cast<char>(c);
    }
    catch (const std::exception &e)
    {
      log_error("Failed to decode character: {}", e.what());
      return std::nullopt;
    }
  };

  if (val.type().is_array() && val.has_operands())
  {
    for (const auto &ch : val.operands())
    {
      if (ch == gen_zero(ch.type()))
        break;

      auto decoded = decode_char(ch);
      if (!decoded)
        return std::nullopt;
      result += *decoded;
    }
  }
  else if (val.is_constant() && val.type().is_signedbv())
  {
    auto decoded = decode_char(val);
    if (!decoded)
      return std::nullopt;
    result += *decoded;
  }
  else
  {
    log_error("Unhandled symbol format in string extraction.");
    return std::nullopt;
  }

  return result;
}

exprt function_call_expr::handle_str_symbol_to_float(const symbolt *sym) const
{
  auto value_opt = extract_string_from_symbol(sym);
  if (!value_opt)
    return from_double(0.0, type_handler_.get_typet("float", 0));

  try
  {
    double dval = std::stod(*value_opt);
    return from_double(dval, type_handler_.get_typet("float", 0));
  }
  catch (const std::exception &e)
  {
    log_error(
      "Failed float conversion from string \"{}\": {}", *value_opt, e.what());
    return from_double(0.0, type_handler_.get_typet("float", 0));
  }
}

exprt function_call_expr::handle_str_symbol_to_int(const symbolt *sym) const
{
  auto value_opt = extract_string_from_symbol(sym);
  if (!value_opt)
    return from_integer(0, type_handler_.get_typet("int", 0));

  const std::string &value = *value_opt;
  if (value.empty() || !std::all_of(value.begin(), value.end(), ::isdigit))
  {
    log_error("Invalid string for integer conversion: \"{}\"", value);
    return from_integer(0, type_handler_.get_typet("int", 0));
  }

  try
  {
    int int_val = std::stoi(value);
    return from_integer(int_val, type_handler_.get_typet("int", 0));
  }
  catch (const std::exception &e)
  {
    log_error("Failed int conversion from string \"{}\": {}", value, e.what());
    return from_integer(0, type_handler_.get_typet("int", 0));
  }
}

const symbolt *
function_call_expr::lookup_python_symbol(const std::string &var_name) const
{
  std::string filename = function_id_.get_filename();
  std::string var_symbol = "py:" + filename + "@" + var_name;
  const symbolt *sym = converter_.find_symbol(var_symbol);

  if (!sym)
  {
    // Don't warn for built-in type constructors (int, float, str, etc.)
    // as they may reference variables from our operational models
    const std::string &func_name = function_id_.get_function();
    if (
      func_name != "int" && func_name != "float" && func_name != "str" &&
      func_name != "bool" && func_name != "bytes")
    {
      log_warning("Symbol not found: {}", var_name);
    }
  }

  return sym;
}

exprt function_call_expr::handle_abs(nlohmann::json &arg) const
{
  // Handle the case where the input is a unary minus applied to a literal
  // (e.g., abs(-5) becomes abs(5)).
  if (arg.contains("_type") && arg["_type"] == "UnaryOp")
  {
    const auto &op = arg["op"];
    const auto &operand = arg["operand"];
    if (op["_type"] == "USub" && operand.contains("value"))
      arg = operand; // Strip the unary minus and use the positive literal
  }

  // Reject strings early
  if (arg.contains("type") && arg["type"] == "str")
    return gen_exception_raise(
      "TypeError", "bad operand type for abs(): 'str'");

  // Also catch string constants without "type" annotation
  if (arg.contains("value") && arg["value"].is_string())
    return gen_exception_raise(
      "TypeError", "bad operand type for abs(): 'str'");

  // If the argument is a numeric literal, evaluate abs() at compile time
  if (arg.contains("value") && arg["value"].is_number())
  {
    if (arg["value"].is_number_integer())
    {
      int value = arg["value"].get<int>();
      arg["value"] = std::abs(value); // Apply abs to integer constant
      arg["type"] = "int";
    }
    else if (arg["value"].is_number_float())
    {
      double value = arg["value"].get<double>();
      arg["value"] = std::abs(value); // Apply abs to float constant
      arg["type"] = "float";
    }

    // Convert the constant into an expression with the appropriate type
    typet t = type_handler_.get_typet(arg["type"], 0);
    exprt expr = converter_.get_expr(arg);
    expr.type() = t;
    return expr;
  }

  // Try to infer type for composite expressions such as BinOp
  if (!arg.contains("type"))
  {
    try
    {
      exprt inferred_expr = converter_.get_expr(arg);
      typet inferred_type = inferred_expr.type();
      exprt abs_expr("abs", inferred_type);
      abs_expr.copy_to_operands(inferred_expr);
      return abs_expr;
    }
    catch (const std::exception &e)
    {
      return gen_exception_raise(
        "TypeError", "failed to infer operand type for abs()");
    }
  }

  // Handle variable references
  if (arg["_type"] == "Name" && arg.contains("id"))
  {
    std::string var_name = arg["id"].get<std::string>();
    const symbolt *sym = lookup_python_symbol(var_name);
    if (sym)
    {
      // Build a symbolic abs() expression with the resolved operand type
      exprt operand_expr = converter_.get_expr(arg);
      typet operand_type = operand_expr.type();

      exprt abs_expr("abs", operand_type);
      abs_expr.copy_to_operands(operand_expr);

      return abs_expr;
    }
    else
    {
      // Variable could not be resolved
      return gen_exception_raise(
        "NameError", "variable '" + var_name + "' is not defined");
    }
  }

  // Final fallback if no type is available
  std::string arg_type = arg.value("type", "");
  if (arg_type.empty())
    return gen_exception_raise(
      "TypeError", "operand to abs() is missing a type");

  // Only numeric types are valid operands for abs()
  if (arg_type != "int" && arg_type != "float" && arg_type != "complex")
    return gen_exception_raise(
      "TypeError", "bad operand type for abs(): '" + arg_type + "'");

  // Fallback for unsupported symbolic expressions (e.g., complex)
  // Currently returns a nil expression to signal unsupported cases
  log_warning("Returning nil expression for abs() with type: {}", arg_type);
  return nil_exprt();
}

exprt function_call_expr::build_constant_from_arg() const
{
  const std::string &func_name = function_id_.get_function();

  // Check if there are no arguments
  if (call_["args"].empty())
  {
    typet t = type_handler_.get_typet(func_name, 0);
    return exprt("constant", t);
  }

  auto arg = call_["args"][0];

  // Handle str(): convert int to str
  if (func_name == "str" && arg["value"].is_number_integer())
    return handle_int_to_str(arg);

  // Handle str(): convert float to str
  else if (func_name == "str" && arg["value"].is_number_float())
    return handle_float_to_str(arg);

  // Handle int(): convert string (from symbol) to int
  else if (func_name == "int" && arg["_type"] == "Name")
  {
    const symbolt *sym = lookup_python_symbol(arg["id"]);
    if (sym && sym->value.is_constant())
      return handle_str_symbol_to_int(sym);
    else
    {
      // Try to get the expression type directly, even if symbol lookup failed
      exprt expr = converter_.get_expr(arg);
      if (expr.type().is_array()) // strings are arrays
      {
        std::string var_name = arg["id"].get<std::string>();
        std::string m = "int() conversion may fail - variable" + var_name +
                        "may contain non-integer string";

        return gen_exception_raise("ValueError", m);
      }
    }
  }

  // Handle int(): convert string literal to int with validation
  else if (
    func_name == "int" && arg.contains("value") && arg["value"].is_string())
  {
    const std::string &str_val = arg["value"].get<std::string>();

    // check if string contains only digits (and optional leading sign)
    bool is_valid = !str_val.empty();
    size_t start_pos = 0;
    if (str_val[0] == '+' || str_val[0] == '-')
    {
      start_pos = 1;
      if (str_val.length() == 1)
        is_valid = false;
    }
    for (size_t i = start_pos; i < str_val.length() && is_valid; i++)
      if (!std::isdigit(str_val[i]))
        is_valid = false;

    if (!is_valid)
      throw std::runtime_error(
        "ValueError: invalid literal for int() with base 10: '" + str_val +
        "'");

    // If valid, convert normally
    int int_val = std::stoi(str_val);
    arg["value"] = int_val;
  }

  size_t arg_size = 1;

  // Handle int(): convert float to int
  if (func_name == "int" && arg["value"].is_number_float())
    handle_float_to_int(arg);

  else if (func_name == "float" && arg["value"].is_string())
  {
    std::string str_val = arg["value"].get<std::string>();

    // Convert to lowercase for case-insensitive comparison
    std::transform(str_val.begin(), str_val.end(), str_val.begin(), ::tolower);

    // Remove whitespace
    str_val.erase(
      std::remove_if(str_val.begin(), str_val.end(), ::isspace), str_val.end());

    // Handle special float string values
    if (str_val == "nan")
    {
      // Create NaN using IEEE float
      ieee_floatt nan_val(ieee_float_spect::double_precision());
      nan_val.make_NaN();
      return nan_val.to_expr();
    }
    else if (
      str_val == "inf" || str_val == "+inf" || str_val == "infinity" ||
      str_val == "+infinity")
    {
      // Create positive infinity
      ieee_floatt inf_val(ieee_float_spect::double_precision());
      inf_val.make_plus_infinity();
      return inf_val.to_expr();
    }
    else if (str_val == "-inf" || str_val == "-infinity")
    {
      // Create negative infinity
      ieee_floatt inf_val(ieee_float_spect::double_precision());
      inf_val.make_minus_infinity();
      return inf_val.to_expr();
    }
    else
    {
      // Try to parse as regular float
      try
      {
        double dval = std::stod(arg["value"].get<std::string>());
        return from_double(dval, type_handler_.get_typet("float", 0));
      }
      catch (const std::exception &e)
      {
        std::string m = "could not convert string to float : " +
                        arg["value"].get<std::string>() + "'";
        return gen_exception_raise("ValueError", m);
      }
    }
  }

  // Handle float(): convert string (from symbol) to float
  else if (func_name == "float" && arg["_type"] == "Name")
  {
    const symbolt *sym = lookup_python_symbol(arg["id"]);
    if (sym && sym->value.is_constant())
      return handle_str_symbol_to_float(sym);
    else
    {
      // Try to get the expression type directly, even if symbol lookup failed
      exprt expr = converter_.get_expr(arg);
      if (expr.type().is_array()) // strings are arrays
      {
        std::string var_name = arg["id"].get<std::string>();
        std::string m = "float() conversion may fail - variable" + var_name +
                        "may contain non-float string";

        return gen_exception_raise("ValueError", m);
      }
    }
  }

  // Handle float(): convert bool to float
  else if (func_name == "float" && arg["value"].is_boolean())
  {
    bool bool_val = arg["value"].get<bool>();
    arg["value"] = bool_val ? 1.0 : 0.0;
  }

  // Handle float(): convert int to float
  else if (func_name == "float" && arg["value"].is_number_integer())
    handle_int_to_float(arg);

  // Handle chr(): convert integer to single-character string
  else if (func_name == "chr")
    return handle_chr(arg);

  // Handle ord(): convert single-character string to integer Unicode code point
  else if (func_name == "ord")
    return handle_ord(arg);

  // Handle hex: Handles hexadecimal string arguments
  else if (func_name == "hex")
    return handle_hex(arg);

  // Handle oct: Handles octal string arguments
  else if (func_name == "oct")
    return handle_oct(arg);

  // Handle abs: Returns the absolute value of an integer or float literal
  else if (func_name == "abs")
    return handle_abs(arg);

  else if (func_name == "str")
    arg_size = handle_str(arg);

  typet t = type_handler_.get_typet(func_name, arg_size);
  exprt expr = converter_.get_expr(arg);

  if (func_name != "str")
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
  else if (subelement["_type"] == "Call")
  {
    obj_name = subelement["func"]["id"];
    if (obj_name == "super")
      obj_name = "self";
  }
  else
    obj_name = subelement["id"].get<std::string>();

  return json_utils::get_object_alias(converter_.ast(), obj_name);
}

bool function_call_expr::is_min_max_call() const
{
  const std::string &func_name = function_id_.get_function();
  return func_name == "min" || func_name == "max";
}

exprt function_call_expr::handle_min_max(
  const std::string &func_name,
  irep_idt comparison_op) const
{
  const auto &args = call_["args"];

  if (args.empty())
    throw std::runtime_error(
      func_name + " expected at least 1 argument, got 0");

  if (args.size() == 1)
    throw std::runtime_error(
      func_name + "() with single iterable argument not yet supported");

  if (args.size() > 2)
    throw std::runtime_error(
      func_name + "() with more than 2 arguments not yet supported");

  // Two arguments case: min/max(a, b)
  exprt arg1 = converter_.get_expr(args[0]);
  exprt arg2 = converter_.get_expr(args[1]);

  // Determine result type (with basic type promotion)
  typet result_type = arg1.type();
  if (!base_type_eq(result_type, arg2.type(), converter_.ns))
  {
    if (result_type.is_signedbv() && arg2.type().is_floatbv())
      result_type = arg2.type(); // Promote to float
    else if (result_type.is_floatbv() && arg2.type().is_signedbv())
      ; // Keep float type
    else
      throw std::runtime_error(
        func_name + "() arguments must be of comparable types: got " +
        result_type.pretty() + " and " + arg2.type().pretty());
  }

  // Create condition: arg1 < arg2 (for min) or arg1 > arg2 (for max)
  exprt condition(comparison_op, type_handler_.get_typet("bool", 0));
  condition.copy_to_operands(arg1, arg2);

  // Create if expression: condition ? arg1 : arg2
  if_exprt result(condition, arg1, arg2);
  result.type() = result_type;

  return result;
}

bool function_call_expr::is_list_method_call() const
{
  if (call_["func"]["_type"] != "Attribute")
    return false;

  const std::string &method_name = function_id_.get_function();

  // Check if this is a known list method
  return method_name == "append" || method_name == "pop" ||
         method_name == "insert" || method_name == "remove" ||
         method_name == "clear" || method_name == "extend";
}

exprt function_call_expr::handle_list_method() const
{
  const std::string &method_name = function_id_.get_function();

  if (method_name == "append")
    return handle_list_append();
  else if (method_name == "pop")
    return handle_list_pop();
  else if (method_name == "insert")
    return handle_list_insert();
  // Add other methods as needed

  throw std::runtime_error("Unsupported list method: " + method_name);
}

exprt function_call_expr::handle_list_append() const
{
  const auto &args = call_["args"];

  if (args.size() != 1)
    throw std::runtime_error("append() takes exactly one argument");

  // Get the list object name
  std::string list_name = get_object_name();

  // Find the list symbol
  symbol_id list_symbol_id = converter_.create_symbol_id();
  list_symbol_id.set_object(list_name);
  const symbolt *list_symbol =
    converter_.find_symbol(list_symbol_id.to_string());

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_name);

  // Get the value to append
  exprt value_to_append = converter_.get_expr(args[0]);
  if (value_to_append.is_constant())
  {
    // Create tmp variable to hold value
    symbolt &append_value_symbol = converter_.create_tmp_symbol(
      call_, "append_value", size_type(), gen_zero(size_type()));
    code_declt append_value(symbol_expr(append_value_symbol));
    append_value.copy_to_operands(value_to_append);
    converter_.current_block->copy_to_operands(append_value);
  }

  converter_.list_type_map[list_symbol->id.as_string()].push_back(
    value_to_append.type());

  return converter_.build_push_list_call(*list_symbol, call_, value_to_append);
}

exprt function_call_expr::handle_list_pop() const
{
  const auto &args = call_["args"];

  // pop() can take 0 or 1 argument (index)
  if (args.size() > 1)
    throw std::runtime_error("pop() takes at most 1 argument");

  std::string list_name = get_object_name();

  symbol_id list_symbol_id = converter_.create_symbol_id();
  list_symbol_id.set_object(list_name);
  const symbolt *list_symbol =
    converter_.find_symbol(list_symbol_id.to_string());

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_name);

  // Create a function call to a dummy pop function that returns an int
  code_function_callt call_expr;
  call_expr.location() = converter_.get_location_from_decl(call_);

  // Create dummy function symbol
  symbolt dummy_func;
  dummy_func.name = "__ESBMC_list_pop_dummy";
  dummy_func.id = dummy_func.name;
  dummy_func.type = code_typet();
  to_code_type(dummy_func.type).return_type() =
    type_handler_.get_typet("int", 0);
  dummy_func.is_extern = true;

  // Add to symbol table if not already present
  auto &symbol_table = converter_.symbol_table();
  if (symbol_table.find_symbol(dummy_func.id) == nullptr)
    symbol_table.add(dummy_func);

  call_expr.function() = symbol_expr(dummy_func);
  call_expr.type() = type_handler_.get_typet("int", 0);

  // Add arguments
  call_expr.arguments().push_back(symbol_expr(*list_symbol));

  if (args.size() == 1)
  {
    exprt index = converter_.get_expr(args[0]);
    call_expr.arguments().push_back(index);
  }

  return call_expr;
}

exprt function_call_expr::handle_list_insert() const
{
  const auto &args = call_["args"];

  if (args.size() != 2)
    throw std::runtime_error("insert() takes exactly 2 arguments");

  std::string list_name = get_object_name();

  symbol_id list_symbol_id = converter_.create_symbol_id();
  list_symbol_id.set_object(list_name);
  const symbolt *list_symbol =
    converter_.find_symbol(list_symbol_id.to_string());

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_name);

  exprt index = converter_.get_expr(args[0]);
  exprt value = converter_.get_expr(args[1]);

  // Create a function call to a dummy insert function
  code_function_callt call_expr;
  call_expr.location() = converter_.get_location_from_decl(call_);

  // Create dummy function symbol
  symbolt dummy_func;
  dummy_func.name = "__ESBMC_list_insert_dummy";
  dummy_func.id = dummy_func.name;
  dummy_func.type = code_typet();
  to_code_type(dummy_func.type).return_type() = empty_typet();
  dummy_func.is_extern = true;

  // Add to symbol table if not already present
  auto &symbol_table = converter_.symbol_table();
  if (symbol_table.find_symbol(dummy_func.id) == nullptr)
    symbol_table.add(dummy_func);

  call_expr.function() = symbol_expr(dummy_func);
  call_expr.type() = empty_typet();

  // Add arguments
  call_expr.arguments().push_back(symbol_expr(*list_symbol));
  call_expr.arguments().push_back(index);
  call_expr.arguments().push_back(value);

  return call_expr;
}

exprt function_call_expr::get()
{
  // Handle non-det functions
  if (is_nondet_call())
  {
    return build_nondet_call();
  }

  // Handle introspection functions
  if (is_introspection_call())
  {
    if (function_id_.get_function() == "isinstance")
      return handle_isinstance();
    else
      return handle_hasattr();
  }

  // Handle input() function
  if (is_input_call())
  {
    return handle_input();
  }

  // Handle min/max functions
  if (is_min_max_call())
  {
    const std::string &func_name = function_id_.get_function();
    if (func_name == "min")
      return handle_min_max("min", exprt::i_lt);
    else
      return handle_min_max("max", exprt::i_gt);
  }

  // Handle list methods
  if (is_list_method_call())
  {
    return handle_list_method();
  }

  const std::string &func_name = function_id_.get_function();

  // Handle math module functions using ESBMC's built-in operations
  if (func_name == "__ESBMC_isnan")
  {
    const auto &args = call_["args"];
    if (args.size() != 1)
      throw std::runtime_error("isnan() expects exactly 1 argument");

    exprt arg_expr = converter_.get_expr(args[0]);

    // Use ESBMC's built-in isnan operation
    exprt isnan_expr("isnan", bool_typet());
    isnan_expr.copy_to_operands(arg_expr);

    return isnan_expr;
  }
  else if (func_name == "__ESBMC_isinf")
  {
    const auto &args = call_["args"];
    if (args.size() != 1)
      throw std::runtime_error("isinf() expects exactly 1 argument");

    exprt arg_expr = converter_.get_expr(args[0]);

    // Use ESBMC's built-in isinf operation
    exprt isinf_expr("isinf", bool_typet());
    isinf_expr.copy_to_operands(arg_expr);

    return isinf_expr;
  }

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
        assert(func_symbol);

        // Update obj attributes from self
        converter_.update_instance_from_self(
          get_classname_from_symbol_id(func_symbol->id.as_string()),
          func_name,
          obj_symbol_id.to_string());
      }
    }
    else
    {
      // Find in global scope
      function_id_.set_class("");
      func_symbol = converter_.find_symbol(function_id_.to_string().c_str());

      if (!func_symbol)
      {
        // Check if this function is defined anywhere in the current Python source
        // by searching the AST directly
        bool is_forward_reference = false;

        is_forward_reference =
          json_utils::search_function_in_ast(converter_.ast(), func_name);

        if (is_forward_reference)
        {
          // Create a forward reference that can be resolved later
          locationt location = converter_.get_location_from_decl(call_);

          code_function_callt call;
          call.location() = location;

          // Create symbol expression for the function (forward reference)
          symbol_exprt func_sym(function_id_.to_string(), code_typet());
          call.function() = func_sym;
          call.type() = empty_typet(); // Will be resolved later

          // Process arguments normally
          for (const auto &arg_node : call_["args"])
          {
            exprt arg = converter_.get_expr(arg_node);
            if (arg.type().is_array())
            {
              if (
                arg_node["_type"] == "Constant" &&
                arg_node["value"].is_string())
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
        else
        {
          // Not a forward reference - use existing behavior for built-ins/imports/undefined functions
          log_warning("Undefined function: {}", func_name.c_str());
          return exprt();
        }
      }
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
    if (converter_.current_lhs)
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

    if (
      function_id_.get_function() == "__ESBMC_get_object_size" &&
      (arg.type() == converter_.get_list_type() ||
       (arg.type().is_pointer() &&
        arg.type().subtype() == converter_.get_list_type())))
    {
      symbolt *list_symbol = converter_.find_symbol(arg.identifier().c_str());
      assert(list_symbol);

      const symbolt *list_size_func_sym =
        converter_.find_symbol("c:list.c@F@list_size");
      assert(list_size_func_sym);

      code_function_callt list_size_func_call;
      list_size_func_call.function() = symbol_expr(*list_size_func_sym);

      // passing arguments to list_size
      if (list_symbol->type.is_pointer())
        list_size_func_call.arguments().push_back(symbol_expr(*list_symbol));
      else
        list_size_func_call.arguments().push_back(
          address_of_exprt(symbol_expr(*list_symbol))); // &l

      // setting return type
      list_size_func_call.type() = size_type();

      return list_size_func_call;
    }

    // Handle function calls used as arguments
    if (arg.is_code() && arg.is_function_call())
    {
      // This is a function call being used as an argument
      // Instead of using the code expression directly, we need to create
      // a side effect expression that represents the function call's result

      // Create a side effect expression for the function call
      side_effect_expr_function_callt func_call;
      func_call.function() = arg.op1(); // The function being called

      // Handle the arguments - op2() is an arguments expression containing operands
      const exprt &args_expr = to_code(arg).op2();
      for (const auto &operand : args_expr.operands())
        func_call.arguments().push_back(operand);

      // Set the type to the return type of the function
      const exprt &func_expr = arg.op1();
      if (func_expr.is_symbol())
      {
        const symbolt *func_symbol =
          converter_.ns.lookup(to_symbol_expr(func_expr));
        if (func_symbol != nullptr)
        {
          const code_typet &func_type = to_code_type(func_symbol->type);
          func_call.type() = func_type.return_type();
        }
      }

      // Use the side effect expression as the argument
      arg = func_call;
    }

    // All array function arguments (e.g. bytes type) are handled as pointers.
    if (arg.type().is_array() || arg.type() == converter_.get_list_type())
    {
      if (arg_node["_type"] == "Constant" && arg_node["value"].is_string())
      {
        arg = string_constantt(
          arg_node["value"].get<std::string>(),
          arg.type(),
          string_constantt::k_default);
      }
      call.arguments().push_back(address_of_exprt(arg));

      if (arg.type() == converter_.get_list_type())
      {
        // Update list element type mapping for function parameters
        const code_typet &type =
          static_cast<const code_typet &>(func_symbol->type);
        const std::string &arg_id =
          type.arguments().at(0).identifier().as_string();
        converter_.list_type_map[arg_id] =
          converter_.list_type_map[arg.identifier().as_string()];
      }
    }
    else
      call.arguments().push_back(arg);
  }

  return std::move(call);
}

exprt function_call_expr::gen_exception_raise(
  std::string exc,
  std::string message) const
{
  if (!type_utils::is_python_exceptions(exc))
  {
    log_error("This exception type is not supported: {}", exc);
    abort();
  }

  typet type = type_handler_.get_typet(exc);

  exprt size = constant_exprt(
    integer2binary(message.size(), bv_width(size_type())),
    integer2string(message.size()),
    size_type());
  typet t = array_typet(char_type(), size);
  string_constantt string_name(message, t, string_constantt::k_default);

  // Construct a constant struct to throw:
  // raise VauleError{ .message=&"Error message" }
  // If the exception model is modified, it might be necessary to make changes
  exprt sym("struct", type);
  sym.copy_to_operands(address_of_exprt(string_name));

  exprt raise = side_effect_exprt("cpp-throw", type);
  raise.move_to_operands(sym);

  return raise;
}
