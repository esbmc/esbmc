#include <python-frontend/function_call_expr.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/python_list.h>
#include <python-frontend/string_builder.h>
#include <util/base_type.h>
#include <util/c_typecast.h>
#include <util/expr_util.h>
#include <util/string_constant.h>
#include <util/arith_tools.h>
#include <util/ieee_float.h>
#include <util/message.h>
#include <util/python_types.h>
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

bool function_call_expr::is_string_arg(const nlohmann::json &arg) const
{
  // Check type annotation
  if (arg.contains("type") && arg["type"] == "str")
    return true;

  // Check if value is a string
  if (arg.contains("value") && arg["value"].is_string())
    return true;

  // Check if it's a string constant
  if (
    arg["_type"] == "Constant" && arg.contains("value") &&
    arg["value"].is_string())
    return true;

  return false;
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

    // Check for nested instance attribute (e.g., self.b.a.method())
    // Exclude module.Class.method() pattern
    bool is_nested_instance_attr = false;
    if (call_["func"]["value"]["_type"] == "Attribute")
    {
      if (call_["func"]["value"]["value"]["_type"] == "Name")
      {
        std::string root_name =
          call_["func"]["value"]["value"]["id"].get<std::string>();
        if (!converter_.is_imported_module(root_name))
        {
          is_nested_instance_attr = true;
        }
      }
    }

    // Handling a function call as a class method call when:
    // (1) The caller corresponds to a class name, for example: MyClass.foo().
    // (2) Calling methods of built-in types, such as int.from_bytes()
    //     All the calls to built-in methods are handled by class methods in operational models.
    // (3) Calling a instance method from a built-in type object, for example: x.bit_length() when x is an int
    // If the caller is a class or a built-in type, the following condition detects a class method call.
    if (
      !is_nested_instance_attr &&
      (is_class(caller, converter_.ast()) ||
       type_utils::is_builtin_type(caller) ||
       type_utils::is_builtin_type(type_handler_.get_var_type(caller))))
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
  const exprt &obj_expr = converter_.get_expr(args[0]);
  const auto &type_arg = args[1];

  auto build_isinstance = [&](const std::string &type_name) {
    typet expected_type = type_handler_.get_typet(type_name, 0);
    exprt t;
    if (expected_type.is_symbol())
    {
      // struct type
      const symbolt *symbol = converter_.ns.lookup(expected_type);
      t = symbol_expr(*symbol);
    }
    else
      t = gen_zero(expected_type);

    exprt isinstance("isinstance", typet("bool"));
    isinstance.copy_to_operands(obj_expr);
    isinstance.move_to_operands(t);
    return isinstance;
  };

  if (type_arg["_type"] == "Name")
  {
    // isinstance(v, int)
    std::string type_name = args[1]["id"];
    return build_isinstance(type_name);
  }
  else if (type_arg["_type"] == "Constant")
  {
    // isintance(v, None)
    std::string type_name = "NoneType";
    return build_isinstance(type_name);
  }
  else if (type_arg["_type"] == "Tuple")
  {
    // isinstance(v, (int, str))
    // converted into instance(v, int) || isinstance(v, str)
    const auto &elts = type_arg["elts"];

    std::string first_type = elts[0]["id"];
    exprt tupe_instance = build_isinstance(first_type);

    for (size_t i = 1; i < elts.size(); ++i)
    {
      if (elts[i]["_type"] != "Name")
        throw std::runtime_error("Unsupported type in isinstance()");

      std::string type_name = elts[i]["id"];
      exprt next_isinstance = build_isinstance(type_name);

      exprt or_expr("or", typet("bool"));
      or_expr.move_to_operands(tupe_instance);
      or_expr.move_to_operands(next_isinstance);
      tupe_instance = or_expr;
    }

    return tupe_instance;
  }
  else
    throw std::runtime_error("Unsupported type format in isinstance()");
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
  typet t = type_handler_.get_typet("str", str_val.size() + 1);
  return converter_.make_char_array_expr(
    std::vector<uint8_t>(str_val.begin(), str_val.end()), t);
}

exprt function_call_expr::handle_float_to_str(nlohmann::json &arg) const
{
  std::string str_val = std::to_string(arg["value"].get<double>());

  // Remove unnecessary trailing zeros and dot if needed (to match Python str behavior)
  // Example: "5.500000" → "5.5"
  str_val.erase(str_val.find_last_not_of('0') + 1, std::string::npos);
  if (str_val.back() == '.')
    str_val.pop_back();

  typet t = type_handler_.get_typet("str", str_val.size() + 1);
  return converter_.make_char_array_expr(
    std::vector<uint8_t>(str_val.begin(), str_val.end()), t);
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
    {
      // Runtime variable: create expression without compile-time evaluation
      exprt var_expr = converter_.get_expr(arg);

      // Since we can't evaluate at compile time, just convert the variable
      // The type should already be correct from get_expr
      return var_expr;
    }

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

exprt function_call_expr::handle_base_conversion(
  nlohmann::json &arg,
  const std::string &func_name,
  const std::string &prefix,
  std::ios_base &(*base_formatter)(std::ios_base &)) const
{
  long long int_value = 0;
  bool is_negative = false;

  // Extract integer value from argument
  if (arg.contains("_type") && arg["_type"] == "UnaryOp")
  {
    const auto &op = arg["op"];
    const auto &operand = arg["operand"];

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
    {
      return gen_exception_raise(
        "TypeError", "Unsupported UnaryOp in " + func_name + "()");
    }
  }
  else if (arg.contains("value") && arg["value"].is_number_integer())
  {
    int_value = arg["value"].get<long long>();
    if (int_value < 0)
      is_negative = true;
  }
  else
  {
    return gen_exception_raise(
      "TypeError", func_name + "() argument must be an integer");
  }

  // Convert to string with appropriate base and prefix
  std::ostringstream oss;
  oss << (is_negative ? "-" + prefix : prefix) << base_formatter;

  // For hex, also apply nouppercase
  if (func_name == "hex")
    oss << std::nouppercase;

  oss << std::llabs(int_value);
  const std::string result_str = oss.str();

  // Create string type and return character array expression
  typet t = type_handler_.get_typet("str", result_str.size() + 1);
  std::vector<uint8_t> string_literal(result_str.begin(), result_str.end());
  return converter_.make_char_array_expr(string_literal, t);
}

exprt function_call_expr::handle_hex(nlohmann::json &arg) const
{
  return handle_base_conversion(arg, "hex", "0x", std::hex);
}

exprt function_call_expr::handle_oct(nlohmann::json &arg) const
{
  return handle_base_conversion(arg, "oct", "0o", std::oct);
}

exprt function_call_expr::handle_ord(nlohmann::json &arg) const
{
  int code_point = 0;

  // Ensure the argument is a string
  if (is_string_arg(arg))
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
  catch (const std::invalid_argument &)
  {
    log_error(
      "Failed float conversion from string \"{}\": invalid argument",
      *value_opt);
    return from_double(0.0, type_handler_.get_typet("float", 0));
  }
  catch (const std::out_of_range &)
  {
    log_error(
      "Failed float conversion from string \"{}\": out of range", *value_opt);
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
  if (is_string_arg(arg))
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
    // Special handling for str() with no arguments: return empty string
    if (func_name == "str")
      return converter_.get_string_builder().build_string_literal("");

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
      if (type_utils::is_string_type(expr.type()))
      {
        std::string var_name = arg["id"].get<std::string>();
        std::string m = "int() conversion may fail - variable" + var_name +
                        "may contain non-integer string";

        return gen_exception_raise("ValueError", m);
      }
    }
  }

  // Handle int(): convert string literal to int with validation
  else if (func_name == "int" && is_string_arg(arg))
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

  else if (func_name == "float" && is_string_arg(arg))
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
      catch (const std::invalid_argument &)
      {
        std::string m = "could not convert string to float : '" +
                        arg["value"].get<std::string>() + "'";
        return gen_exception_raise("ValueError", m);
      }
      catch (const std::out_of_range &)
      {
        std::string m = "could not convert string to float : '" +
                        arg["value"].get<std::string>() + "' (out of range)";
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
      if (type_utils::is_string_type(expr.type()))
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
  {
    /* For attribute chains, use the class name resolved by build_function_id()
     * 
     * When we have self.f.foo(), the function ID builder has already determined
     * that f's type is Foo and stored it in function_id_. We reuse that result
     * rather than re-extracting "f" which would be incorrect.
     */
    if (!function_id_.get_class().empty())
    {
      std::string class_name = function_id_.get_class();
      obj_name =
        (class_name.find("tag-") == 0) ? class_name.substr(4) : class_name;
    }
    else
    {
      obj_name = subelement["attr"].get<std::string>();
    }
  }
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

exprt function_call_expr::handle_list_insert() const
{
  const auto &args = call_["args"];

  if (args.size() != 2)
    throw std::runtime_error("insert() takes exactly two arguments");

  std::string list_name = get_object_name();

  symbol_id list_symbol_id = converter_.create_symbol_id();
  list_symbol_id.set_object(list_name);
  const symbolt *list_symbol =
    converter_.find_symbol(list_symbol_id.to_string());

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_name);

  exprt index_expr = converter_.get_expr(args[0]);
  exprt value_to_insert = converter_.get_expr(args[1]);

  if (value_to_insert.is_constant())
  {
    symbolt &insert_value_symbol = converter_.create_tmp_symbol(
      call_, "insert_value", size_type(), gen_zero(size_type()));
    code_declt insert_value(symbol_expr(insert_value_symbol));
    insert_value.copy_to_operands(value_to_insert);
    converter_.current_block->copy_to_operands(insert_value);
  }

  python_list list(converter_, nlohmann::json());
  list.add_type_info(
    list_symbol->id.as_string(),
    value_to_insert.identifier().as_string(),
    value_to_insert.type());

  return list.build_insert_list_call(
    *list_symbol, index_expr, call_, value_to_insert);
}

exprt function_call_expr::handle_list_clear() const
{
  // Get the list object name
  std::string list_name = get_object_name();

  // Find the list symbol
  symbol_id list_symbol_id = converter_.create_symbol_id();
  list_symbol_id.set_object(list_name);
  const symbolt *list_symbol =
    converter_.find_symbol(list_symbol_id.to_string());

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_name);

  // Find the list_clear C function
  const symbolt *clear_func =
    converter_.symbol_table().find_symbol("c:list.c@F@list_clear");
  if (!clear_func)
    throw std::runtime_error("Clear function symbol not found");

  // Build function call
  code_function_callt clear_call;
  clear_call.function() = symbol_expr(*clear_func);
  clear_call.arguments().push_back(symbol_expr(*list_symbol));
  clear_call.type() = empty_typet();
  clear_call.location() = converter_.get_location_from_decl(call_);

  return clear_call;
}

bool function_call_expr::is_list_method_call() const
{
  if (call_["func"]["_type"] != "Attribute")
    return false;

  const std::string &method_name = function_id_.get_function();

  // Check if this is a known list method
  return method_name == "append" || method_name == "pop" ||
         method_name == "insert" || method_name == "remove" ||
         method_name == "clear" || method_name == "extend" ||
         method_name == "insert";
}

exprt function_call_expr::handle_list_method() const
{
  const std::string &method_name = function_id_.get_function();

  if (method_name == "append")
    return handle_list_append();
  if (method_name == "insert")
    return handle_list_insert();
  if (method_name == "extend")
    return handle_list_extend();
  if (method_name == "clear")
    return handle_list_clear();

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

  python_list list(converter_, nlohmann::json());

  list.add_type_info(
    list_symbol->id.as_string(),
    value_to_append.identifier().as_string(),
    value_to_append.type());

  return list.build_push_list_call(*list_symbol, call_, value_to_append);
}

exprt function_call_expr::handle_list_extend() const
{
  const auto &args = call_["args"];

  if (args.size() != 1)
    throw std::runtime_error("extend() takes exactly one argument");

  std::string list_name = get_object_name();

  symbol_id list_symbol_id = converter_.create_symbol_id();
  list_symbol_id.set_object(list_name);
  const symbolt *list_symbol =
    converter_.find_symbol(list_symbol_id.to_string());

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_name);

  exprt other_list = converter_.get_expr(args[0]);

  python_list list(converter_, nlohmann::json());

  return list.build_extend_list_call(*list_symbol, call_, other_list);
}

bool function_call_expr::is_print_call() const
{
  const std::string &func_name = function_id_.get_function();
  return func_name == "print";
}

exprt function_call_expr::handle_print() const
{
  // Process all arguments to ensure expressions are evaluated
  const auto &args = call_["args"];

  for (const auto &arg_node : args)
  {
    // Evaluate each argument expression
    // This ensures that any side effects or expressions are properly processed
    converter_.get_expr(arg_node);
  }

  // Print doesn't return a value, so return a nil expression
  // This won't affect verification but ensures arguments are evaluated
  return nil_exprt();
}

bool function_call_expr::is_re_module_call() const
{
  const std::string &func_name = function_id_.get_function();

  return (func_name == "match" || func_name == "search" ||
          func_name == "fullmatch") &&
         call_["func"]["_type"] == "Attribute" && get_object_name() == "re";
}

exprt function_call_expr::validate_re_module_args() const
{
  const auto &args = call_["args"];

  for (size_t i = 0; i < std::min(args.size(), size_t(2)); ++i)
  {
    exprt arg_expr = converter_.get_expr(args[i]);
    const typet &arg_type = arg_expr.type();

    if (!type_utils::is_string_type(arg_type))
    {
      std::ostringstream msg;
      msg << "expected string or bytes-like object, got '"
          << type_handler_.type_to_string(arg_type) << "'";
      return gen_exception_raise("TypeError", msg.str());
    }
  }

  return nil_exprt(); // Validation passed
}

bool function_call_expr::is_any_call() const
{
  const std::string &func_name = function_id_.get_function();
  return func_name == "any";
}

exprt function_call_expr::handle_any() const
{
  const auto &args = call_["args"];

  if (args.empty())
    throw std::runtime_error("any() expected at least 1 argument, got 0");

  if (args.size() > 1)
    throw std::runtime_error("any() takes at most 1 argument");

  const auto &arg = args[0];

  if (arg["_type"] != "List")
    throw std::runtime_error("any() currently only supports list literals");

  const auto &elts = arg["elts"];

  // Empty list returns False
  if (elts.empty())
    return gen_boolean(false);

  // Build an OR expression of all elements' truthiness
  exprt result;
  bool first = true;

  for (const auto &elt : elts)
  {
    exprt element = converter_.get_expr(elt);

    // Check if element is truthy
    exprt is_truthy;

    if (element.type() == none_type())
    {
      // None is always falsy
      is_truthy = gen_boolean(false);
    }
    else if (element.type().is_bool())
    {
      // Bool: use directly
      is_truthy = element;
    }
    else if (
      element.type().id() == "signedbv" || element.type().id() == "unsignedbv")
    {
      // Integer: truthy if != 0
      exprt zero = gen_zero(element.type());
      is_truthy = not_exprt(equality_exprt(element, zero));
    }
    else if (element.type().id() == "floatbv")
    {
      // Float: truthy if != 0.0
      exprt zero = gen_zero(element.type());
      is_truthy = not_exprt(equality_exprt(element, zero));
    }
    else if (element.type().is_pointer())
    {
      // Pointer: truthy if not NULL
      exprt null_ptr = gen_zero(element.type());
      is_truthy = not_exprt(equality_exprt(element, null_ptr));
    }
    else
    {
      // For other types, assume truthy (conservative)
      is_truthy = gen_boolean(true);
    }

    // OR with accumulated result
    if (first)
    {
      result = is_truthy;
      first = false;
    }
    else
      result = or_exprt(result, is_truthy);
  }

  return result;
}

std::vector<function_call_expr::FunctionHandler>
function_call_expr::get_dispatch_table()
{
  return {
    // Print function
    {[this]() { return is_print_call(); },
     [this]() { return handle_print(); },
     "print()"},

    // Non-deterministic functions
    {[this]() { return is_nondet_call(); },
     [this]() { return build_nondet_call(); },
     "nondet functions"},

    // Introspection functions (isinstance, hasattr)
    {[this]() { return is_introspection_call(); },
     [this]() {
       if (function_id_.get_function() == "isinstance")
         return handle_isinstance();
       else
         return handle_hasattr();
     },
     "isinstance/hasattr"},

    // Input function
    {[this]() { return is_input_call(); },
     [this]() { return handle_input(); },
     "input()"},

    // Any function
    {[this]() { return is_any_call(); },
     [this]() { return handle_any(); },
     "any()"},

    // Min/Max functions
    {[this]() { return is_min_max_call(); },
     [this]() {
       const std::string &func_name = function_id_.get_function();
       if (func_name == "min")
         return handle_min_max("min", exprt::i_lt);
       else
         return handle_min_max("max", exprt::i_gt);
     },
     "min/max"},

    // List methods
    {[this]() { return is_list_method_call(); },
     [this]() { return handle_list_method(); },
     "list methods"},

    // Math module functions
    {[this]() {
       const std::string &func_name = function_id_.get_function();
       return func_name == "__ESBMC_isnan" || func_name == "__ESBMC_isinf";
     },
     [this]() {
       const std::string &func_name = function_id_.get_function();
       const auto &args = call_["args"];

       if (func_name == "__ESBMC_isnan")
       {
         if (args.size() != 1)
           throw std::runtime_error("isnan() expects exactly 1 argument");

         exprt arg_expr = converter_.get_expr(args[0]);
         exprt isnan_expr("isnan", bool_typet());
         isnan_expr.copy_to_operands(arg_expr);
         return isnan_expr;
       }
       else // __ESBMC_isinf
       {
         if (args.size() != 1)
           throw std::runtime_error("isinf() expects exactly 1 argument");

         exprt arg_expr = converter_.get_expr(args[0]);
         exprt isinf_expr("isinf", bool_typet());
         isinf_expr.copy_to_operands(arg_expr);
         return isinf_expr;
       }
     },
     "isnan/isinf"},

    // Math module functions
    {[this]() {
       const std::string &func_name = function_id_.get_function();
       bool is_math_module = false;
       if (call_["func"]["_type"] == "Attribute")
       {
         std::string caller = get_object_name();
         is_math_module = (caller == "math");
       }
       return is_math_module && func_name == "sqrt";
     },
     [this]() {
       const auto &args = call_["args"];
       if (args.size() != 1)
         throw std::runtime_error("sqrt() expects exactly 1 argument");

       exprt arg_expr = converter_.get_expr(args[0]);

       // Promote to float if needed
       exprt double_operand = arg_expr;
       if (!arg_expr.type().is_floatbv())
       {
         double_operand =
           exprt("typecast", type_handler_.get_typet("float", 0));
         double_operand.copy_to_operands(arg_expr);
       }

       // Create domain check: x < 0 (error condition)
       exprt zero = gen_zero(type_handler_.get_typet("float", 0));
       exprt domain_check = exprt("<", type_handler_.get_typet("bool", 0));
       domain_check.copy_to_operands(double_operand, zero);

       // Create exception for domain violation
       exprt raise = gen_exception_raise("ValueError", "math domain error");

       // Add location information
       locationt loc = converter_.get_location_from_decl(call_);
       raise.location() = loc;
       raise.location().user_provided(true);

       // Call python_math to handle the actual sqrt call
       exprt sqrt_result =
         converter_.get_math_handler().handle_sqrt(arg_expr, call_);

       // Return conditional: if (x < 0) raise ValueError else sqrt(x)
       if_exprt conditional(domain_check, raise, sqrt_result);
       conditional.type() = type_handler_.get_typet("float", 0);

       return conditional;
     },
     "math.sqrt()"},

    // Built-in type constructors (int, float, str, bool, etc.)
    {[this]() {
       const std::string &func_name = function_id_.get_function();
       return type_utils::is_builtin_type(func_name) ||
              type_utils::is_consensus_type(func_name);
     },
     [this]() { return build_constant_from_arg(); },
     "built-in type constructors"},

    // Regex module validation
    {[this]() { return is_re_module_call(); },
     [this]() {
       exprt validation_result = validate_re_module_args();
       if (!validation_result.is_nil())
         return validation_result;

       // If validation passes, handle as general function call
       return handle_general_function_call();
     },
     "re module functions"}};
}

exprt function_call_expr::get()
{
  // Use dispatch table to handle special function types
  auto dispatch_table = get_dispatch_table();

  for (const auto &handler : dispatch_table)
  {
    if (handler.predicate())
      return handler.handler();
  }

  // General function call handling
  return handle_general_function_call();
}

exprt function_call_expr::handle_general_function_call()
{
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
  const symbolt *func_symbol = converter_.find_symbol(func_symbol_id);

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
        function_id_.get_function(),
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
        if (obj_symbol && func_symbol)
        {
          converter_.update_instance_from_self(
            get_classname_from_symbol_id(func_symbol->id.as_string()),
            function_id_.get_function(),
            obj_symbol_id.to_string());
        }

        // Handle forward reference: method not yet in symbol table
        if (!func_symbol)
        {
          std::string class_name = function_id_.get_class();
          std::string method_name = function_id_.get_function();

          // Find all possible class types for this object
          std::vector<std::string> possible_classes =
            find_possible_class_types(obj_symbol);

          // If no classes found, use the inferred class name
          if (possible_classes.empty())
            possible_classes.push_back(class_name);

          // Check if method exists in any of the possible classes
          bool method_exists = false;
          for (const auto &check_class : possible_classes)
          {
            if (method_exists_in_class_hierarchy(check_class, method_name))
            {
              method_exists = true;
              break;
            }
          }

          // Only report AttributeError if:
          // 1. Method doesn't exist in the class hierarchy, AND
          // 2. We're not currently processing any method in the same class
          //    (methods may reference other methods not yet in the symbol table)
          std::string current_func = converter_.current_function_name();
          bool is_in_same_class = false;

          // Check if we're in any method of the target class
          for (const auto &check_class : possible_classes)
          {
            // We're in the same class if:
            // - current function is the class name (constructor)
            // - current function is __init__
            // - the function symbol exists and contains the class marker for this class
            if (current_func == check_class || current_func == "__init__")
            {
              is_in_same_class = true;
              break;
            }

            // Check if current function belongs to this class by looking for @C@ClassName pattern
            std::string class_marker = std::string(CLASS_MARKER) + check_class +
                                       std::string(FUNCTION_MARKER);
            const symbolt *current_func_sym =
              converter_.find_symbol(converter_.create_symbol_id().to_string());
            if (
              current_func_sym && current_func_sym->id.as_string().find(
                                    class_marker) != std::string::npos)
            {
              is_in_same_class = true;
              break;
            }
          }

          if (!method_exists && !is_in_same_class)
          {
            // Generate AttributeError
            return generate_attribute_error(method_name, possible_classes);
          }

          // Method exists or we're in a constructor - create forward reference
          locationt location = converter_.get_location_from_decl(call_);
          code_function_callt call;
          call.location() = location;
          call.function() = symbol_exprt(func_symbol_id, code_typet());
          call.type() = empty_typet();

          if (obj_symbol)
            call.arguments().push_back(
              gen_address_of(symbol_expr(*obj_symbol)));

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

          return call;
        }
      }
    }
    else
    {
      // Find in global scope
      function_id_.set_class("");
      func_symbol = converter_.find_symbol(function_id_.to_string());

      if (!func_symbol)
      {
        // Check if this function is defined anywhere in the current Python source
        // by searching the AST directly
        bool is_forward_reference = false;

        is_forward_reference = json_utils::search_function_in_ast(
          converter_.ast(), function_id_.get_function());

        if (is_forward_reference)
        {
          // Create a forward reference that can be resolved later
          locationt location = converter_.get_location_from_decl(call_);

          code_function_callt call;
          call.location() = location;

          // Create symbol expression for the function (forward reference)
          symbol_exprt func_sym(function_id_.to_string(), code_typet());
          call.function() = func_sym;

          // Extract return type from function definition in AST
          typet return_type = empty_typet();
          const auto &func_node = find_function(
            converter_.ast()["body"], function_id_.get_function());
          if (!func_node.empty())
          {
            if (
              func_node.contains("returns") && !func_node["returns"].is_null())
            {
              const auto &returns = func_node["returns"];
              if (returns.contains("id"))
              {
                return_type =
                  type_handler_.get_typet(returns["id"].get<std::string>());
              }
            }
            exprt body = converter_.get_block(func_node["body"]);
            exprt const_return = converter_.get_function_constant_return(body);
            if (!const_return.is_nil())
              return const_return;
          }

          call.type() = return_type;

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

          return call;
        }
        else
        {
          const std::string &func_name = function_id_.get_function();
          log_warning(
            "Undefined function '{}' - replacing with assert(false)",
            func_name);
          return gen_unsupported_function_assert(func_name);
        }
      }
    }
  }

  if (func_symbol != nullptr)
  {
    exprt type_error = check_argument_types(func_symbol, call_["args"]);
    if (!type_error.is_nil())
      return type_error;
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
    call.type() = type_handler_.get_typet(func_symbol->name.as_string());
    // Self is the LHS
    if (converter_.current_lhs)
      call.arguments().push_back(gen_address_of(*converter_.current_lhs));
  }
  else if (function_type_ == FunctionType::InstanceMethod)
  {
    if (obj_symbol)
    {
      call.arguments().push_back(gen_address_of(symbol_expr(*obj_symbol)));
    }
    else
    {
      // Nested attribute: build expression dynamically
      if (
        call_["func"]["_type"] == "Attribute" &&
        call_["func"].contains("value"))
      {
        exprt obj_expr = converter_.get_expr(call_["func"]["value"]);
        call.arguments().push_back(gen_address_of(obj_expr));
      }
      else
      {
        assert(
          false &&
          "InstanceMethod requires obj_symbol or valid attribute chain");
      }
    }
  }
  else if (function_type_ == FunctionType::ClassMethod)
  {
    // Check if this is an instance method being called through the class
    // e.g., MyClass.method(instance) where the first param should be 'self'
    const code_typet &func_type = to_code_type(func_symbol->type);
    bool first_param_is_self = false;

    if (!func_type.arguments().empty())
    {
      const std::string &first_param =
        func_type.arguments()[0].get_base_name().as_string();
      first_param_is_self = (first_param == "self");
    }

    // If first parameter is 'self' and we have positional arguments,
    // the first positional arg should be treated as 'self', not as a regular argument
    if (
      first_param_is_self && !call_["args"].empty() &&
      (!call_.contains("keywords") || call_["keywords"].empty() ||
       call_["keywords"][0]["arg"] != "self"))
    {
      // First positional argument will be added in the loop below as 'self'
      // Don't add a NULL cls parameter
    }
    else
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
        // Handling function call from binary expressions such as: (x+1).bit_length()
        call.arguments().push_back(converter_.get_expr(call_["func"]["value"]));
      }
    }
  }

  for (const auto &arg_node : call_["args"])
  {
    exprt arg = converter_.get_expr(arg_node);

    if (
      function_id_.get_function() == "__ESBMC_get_object_size" &&
      (arg.type() == type_handler_.get_list_type() ||
       (arg.type().is_pointer() &&
        arg.type().subtype() == type_handler_.get_list_type())))
    {
      symbolt *list_symbol =
        converter_.find_symbol(arg.identifier().as_string());
      assert(list_symbol);

      const symbolt *list_size_func_sym =
        converter_.find_symbol("c:list.c@F@list_size");
      assert(list_size_func_sym);

      code_function_callt list_size_func_call;
      list_size_func_call.function() = symbol_expr(*list_size_func_sym);

      // passing arguments to list_size
      list_size_func_call.arguments().push_back(symbol_expr(*list_symbol));

      // setting return type
      list_size_func_call.type() = signedbv_typet(64);

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
          typet return_type = func_type.return_type();

          // Special handling for constructors
          if (return_type.id() == "constructor")
          {
            // For constructors, use the class type instead of "constructor"
            return_type =
              type_handler_.get_typet(func_symbol->name.as_string());
          }

          func_call.type() = return_type;
        }
      }

      // Use the side effect expression as the argument
      arg = func_call;
    }

    if (arg.type() == type_handler_.get_list_type())
    {
      // Update list element type mapping for function parameters
      const code_typet &type =
        static_cast<const code_typet &>(func_symbol->type);
      const std::string &arg_id =
        type.arguments().at(0).identifier().as_string();

      python_list::copy_type_info(arg.identifier().as_string(), arg_id);
    }

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

  return call;
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

codet function_call_expr::gen_unsupported_function_assert(
  const std::string &func_name) const
{
  locationt location = converter_.get_location_from_decl(call_);
  std::string message = "Unsupported function '" + func_name + "' is reached";
  location.user_provided(true);
  location.comment(message);

  exprt false_expr = gen_boolean(false);
  code_assertt assert_code(false_expr);
  assert_code.location() = location;

  return assert_code;
}

std::vector<std::string>
function_call_expr::find_possible_class_types(const symbolt *obj_symbol) const
{
  std::vector<std::string> possible_classes;

  if (!obj_symbol)
    return possible_classes;

  typet obj_type = obj_symbol->type;
  if (obj_type.is_pointer())
    obj_type = obj_type.subtype();
  if (obj_type.id() == "symbol")
    obj_type = converter_.ns.follow(obj_type);

  // If type is a struct, extract the class name from the struct tag
  if (obj_type.is_struct())
  {
    const struct_typet &struct_type = to_struct_type(obj_type);
    std::string tag = struct_type.tag().as_string();
    std::string actual_class = (tag.find("tag-") == 0) ? tag.substr(4) : tag;
    possible_classes.push_back(actual_class);
    return possible_classes;
  }

  // Type is a primitive (e.g., floatbv) - trace through AST to find actual types
  std::string var_name = obj_symbol->name.as_string();
  nlohmann::json var_decl = json_utils::find_var_decl(
    var_name, converter_.current_function_name(), converter_.ast());

  if (var_decl.empty() || !var_decl.contains("value"))
    return possible_classes;

  const auto &value = var_decl["value"];

  // Check if assigned from a function call
  if (value["_type"] != "Call" || value["func"]["_type"] != "Name")
    return possible_classes;

  std::string func_name = value["func"]["id"].get<std::string>();

  // Look up the function definition
  const auto &func_node =
    json_utils::find_function(converter_.ast()["body"], func_name);

  if (
    func_node.empty() || !func_node.contains("returns") ||
    func_node["returns"].is_null())
    return possible_classes;

  const auto &returns = func_node["returns"];
  if (returns["_type"] != "Name" || !returns.contains("id"))
    return possible_classes;

  std::string return_type = returns["id"].get<std::string>();

  // If return type is 'Any', analyze the function body to find actual return classes
  if (return_type == "Any" && func_node.contains("body"))
  {
    std::function<void(const nlohmann::json &)> find_returns;
    find_returns = [&](const nlohmann::json &node) {
      if (!node.is_object())
        return;

      std::string node_type = node["_type"].get<std::string>();

      if (node_type == "Return" && node.contains("value"))
      {
        const auto &ret_val = node["value"];
        if (ret_val["_type"] == "Call" && ret_val["func"]["_type"] == "Name")
        {
          std::string class_name = ret_val["func"]["id"].get<std::string>();
          if (json_utils::is_class(class_name, converter_.ast()))
            possible_classes.push_back(class_name);
        }
      }
      else if (node_type == "If")
      {
        // Check both branches
        if (node.contains("body"))
          for (const auto &stmt : node["body"])
            find_returns(stmt);
        if (node.contains("orelse"))
          for (const auto &stmt : node["orelse"])
            find_returns(stmt);
      }
    };

    for (const auto &stmt : func_node["body"])
      find_returns(stmt);
  }

  return possible_classes;
}

bool function_call_expr::method_exists_in_class_hierarchy(
  const std::string &class_name,
  const std::string &method_name) const
{
  const auto &class_node =
    json_utils::find_class(converter_.ast()["body"], class_name);

  if (class_node.empty())
    return false;

  // Check if method exists in this class
  if (json_utils::search_function_in_ast(class_node["body"], method_name))
    return true;

  // Check base classes
  if (class_node.contains("bases"))
  {
    for (const auto &base : class_node["bases"])
    {
      if (base.contains("id"))
      {
        std::string base_name = base["id"].get<std::string>();
        if (method_exists_in_class_hierarchy(base_name, method_name))
          return true;
      }
    }
  }

  return false;
}

exprt function_call_expr::generate_attribute_error(
  const std::string &method_name,
  const std::vector<std::string> &possible_classes) const
{
  locationt location = converter_.get_location_from_decl(call_);
  std::ostringstream error_msg;

  if (possible_classes.size() > 1)
  {
    // Multiple possible classes
    std::string display_classes;
    for (size_t i = 0; i < possible_classes.size(); ++i)
    {
      if (i > 0)
        display_classes += ", ";
      display_classes += "'" + possible_classes[i] + "'";
    }
    error_msg << "AttributeError: object has no attribute '" << method_name
              << "' (possible types: " << display_classes << ")";
  }
  else if (possible_classes.size() == 1)
  {
    error_msg << "AttributeError: '" << possible_classes[0]
              << "' object has no attribute '" << method_name << "'";
  }
  else
  {
    error_msg << "AttributeError: object has no attribute '" << method_name
              << "'";
  }

  log_warning("{}", error_msg.str());

  code_assertt assert_code(gen_boolean(false));
  assert_code.location() = location;
  assert_code.location().user_provided(true);
  assert_code.location().comment(error_msg.str());

  return assert_code;
}

exprt function_call_expr::check_argument_types(
  const symbolt *func_symbol,
  const nlohmann::json &args) const
{
  // Only perform type checking if --strict-types is enabled
  if (!config.options.get_bool_option("strict-types"))
    return nil_exprt();

  const code_typet &func_type = to_code_type(func_symbol->type);
  const auto &params = func_type.arguments();

  // Determine parameter offset based on actual function signature
  size_t param_offset = 0;

  if (function_type_ == FunctionType::InstanceMethod)
  {
    // Instance methods always have 'self' as first parameter
    param_offset = 1;
  }
  else if (function_type_ == FunctionType::ClassMethod)
  {
    // For class methods, check if first parameter is actually 'self' or 'cls'
    // Static methods are called as ClassMethod but don't have self/cls
    if (!params.empty())
    {
      const std::string &first_param = params[0].get_base_name().as_string();
      if (first_param == "self" || first_param == "cls")
        param_offset = 1;
      // Otherwise it's a static method, param_offset stays 0
    }
  }

  for (size_t i = 0; i < args.size(); ++i)
  {
    size_t param_idx = i + param_offset;
    if (param_idx >= params.size())
      break;

    exprt arg = converter_.get_expr(args[i]);
    const typet &expected_type = params[param_idx].type();
    const typet &actual_type = arg.type();

    // Check for type mismatch
    if (!base_type_eq(expected_type, actual_type, converter_.ns))
    {
      std::string expected_str = type_handler_.type_to_string(expected_type);
      std::string actual_str = type_handler_.type_to_string(actual_type);

      std::ostringstream msg;
      msg << "TypeError: Argument " << (i + 1) << " has incompatible type '"
          << actual_str << "'; expected '" << expected_str << "'";

      exprt exception = gen_exception_raise("TypeError", msg.str());

      // Add location information from the call
      locationt loc = converter_.get_location_from_decl(call_);
      exception.location() = loc;
      exception.location().user_provided(true);

      return exception;
    }
  }

  return nil_exprt();
}
