#include <python-frontend/function_call_expr.h>
#include <python-frontend/exception_utils.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/python_exception_handler.h>
#include <python-frontend/python_list.h>
#include <python-frontend/string_builder.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/c_typecast.h>
#include <util/expr_util.h>
#include <util/ieee_float.h>
#include <util/message.h>
#include <util/python_types.h>
#include <util/std_expr.h>
#include <util/string_constant.h>

#include <regex>
#include <stdexcept>

using namespace json_utils;
namespace
{
// Constants for input handling
constexpr int DEFAULT_NONDET_STR_LENGTH = 16;

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
    R"(nondet_(int|char|bool|float|str)|__VERIFIER_nondet_(int|char|bool|float|str))");

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

static int get_nondet_str_length()
{
  std::string opt_value = config.options.get_option("nondet-str-length");
  if (!opt_value.empty())
  {
    try
    {
      int length = std::stoi(opt_value);
      if (length > 0)
        return length;
    }
    catch (...)
    {
    }
  }
  return DEFAULT_NONDET_STR_LENGTH;
}

exprt function_call_expr::handle_input() const
{
  // input() returns a non-deterministic string
  // Model as a bounded C-string without embedded nulls.
  int max_str_length = get_nondet_str_length();
  typet string_type = type_handler_.get_typet("str", max_str_length);

  symbolt &input_sym =
    converter_.create_tmp_symbol(call_, "$input_str$", string_type, exprt());
  code_declt decl(symbol_expr(input_sym));
  decl.location() = converter_.get_location_from_decl(call_);
  converter_.add_instruction(decl);

  exprt nondet_value("sideeffect", string_type);
  nondet_value.statement("nondet");
  code_assignt nondet_assign(symbol_expr(input_sym), nondet_value);
  nondet_assign.location() = converter_.get_location_from_decl(call_);
  converter_.add_instruction(nondet_assign);

  symbolt &len_sym =
    converter_.create_tmp_symbol(call_, "$input_len$", size_type(), exprt());
  code_declt len_decl(symbol_expr(len_sym));
  len_decl.location() = converter_.get_location_from_decl(call_);
  converter_.add_instruction(len_decl);

  exprt len_nondet("sideeffect", size_type());
  len_nondet.statement("nondet");
  code_assignt len_assign(symbol_expr(len_sym), len_nondet);
  len_assign.location() = converter_.get_location_from_decl(call_);
  converter_.add_instruction(len_assign);

  exprt len_bound("<", bool_type());
  len_bound.copy_to_operands(
    symbol_expr(len_sym), from_integer(max_str_length, size_type()));
  codet assume_len("assume");
  assume_len.copy_to_operands(len_bound);
  assume_len.location() = converter_.get_location_from_decl(call_);
  converter_.add_instruction(assume_len);

  index_exprt term_pos(
    symbol_expr(input_sym), symbol_expr(len_sym), char_type());
  code_assignt term_assign(term_pos, from_integer(0, char_type()));
  term_assign.location() = converter_.get_location_from_decl(call_);
  converter_.add_instruction(term_assign);

  return symbol_expr(input_sym);
}

exprt function_call_expr::build_nondet_call() const
{
  const std::string &func_name = function_id_.get_function();

  // Function name pattern: nondet_(type). e.g: nondet_bool(), nondet_int()
  size_t underscore_pos = func_name.rfind("_");
  std::string type = func_name.substr(underscore_pos + 1);

  if (type == "str")
  {
    int max_str_length = get_nondet_str_length();

    typet char_array_type =
      array_typet(char_type(), from_integer(max_str_length, size_type()));

    // Create a temporary variable to hold the nondeterministic string
    symbolt &nondet_str_symbol = converter_.create_tmp_symbol(
      call_, "$nondet_str$", char_array_type, exprt());

    // Declare the temporary
    code_declt decl(symbol_expr(nondet_str_symbol));
    decl.location() = converter_.get_location_from_decl(call_);
    converter_.add_instruction(decl);

    // Create nondet assignment for the array
    exprt nondet_value("sideeffect", char_array_type);
    nondet_value.statement("nondet");

    code_assignt nondet_assign(symbol_expr(nondet_str_symbol), nondet_value);
    nondet_assign.location() = converter_.get_location_from_decl(call_);
    converter_.add_instruction(nondet_assign);

    // Ensure null terminator at the last position
    exprt last_index = from_integer(max_str_length - 1, size_type());
    exprt null_char = from_integer(0, char_type());

    index_exprt last_elem(symbol_expr(nondet_str_symbol), last_index);
    code_assignt null_assign(last_elem, null_char);
    null_assign.location() = converter_.get_location_from_decl(call_);
    converter_.add_instruction(null_assign);

    // Return address of first element: &arr[0] which is char*
    index_exprt first_elem(
      symbol_expr(nondet_str_symbol), from_integer(0, size_type()));
    return address_of_exprt(first_elem);
  }

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
  const auto &obj_arg = args[0];
  const auto &type_arg = args[1];

  // Check if the first argument is a type object (e.g., x = int; isinstance(x, str))
  // Type objects themselves are not instances of other types (except 'type')
  if (obj_arg["_type"] == "Name")
  {
    const std::string &obj_name = obj_arg["id"];

    // Check if this variable holds a type object by checking the symbol
    std::string lookup_name = obj_name;
    if (obj_expr.is_symbol())
    {
      const symbol_exprt &sym_expr = to_symbol_expr(obj_expr);
      lookup_name = sym_expr.get_identifier().as_string();
    }

    const symbolt *var_symbol = converter_.ns.lookup(lookup_name);
    if (var_symbol && var_symbol->value.is_constant())
    {
      const constant_exprt &const_val = to_constant_expr(var_symbol->value);
      std::string value_str = const_val.get_value().as_string();
      // Check if this constant value is a type name
      if (type_utils::is_type_identifier(value_str))
      {
        auto extract_type_name = [](const nlohmann::json &node) -> std::string {
          const std::string node_type = node["_type"];
          if (node_type == "Name")
            return node["id"];
          return "";
        };
        std::string type_name = extract_type_name(type_arg);
        if (type_name == "type")
          return true_exprt();
        else
          return false_exprt();
      }
    }
  }

  // Extract type name from various AST node formats
  auto extract_type_name = [](const nlohmann::json &node) -> std::string {
    const std::string node_type = node["_type"];

    if (node_type == "Name")
    {
      // isinstance(v, int)
      return node["id"];
    }
    else if (node_type == "Constant")
    {
      // isinstance(v, None)
      return "NoneType";
    }
    else if (node_type == "Attribute")
    {
      // isinstance(v, MyClass.InnerClass)
      return node["attr"];
    }
    else if (node_type == "Call")
    {
      // isinstance(v, type(None)) or isinstance(v, type(x))
      const auto &func = node["func"];

      if (func["_type"] != "Name" || func["id"] != "type")
        throw std::runtime_error(
          "Only type() calls are supported in isinstance()");

      const auto &call_args = node["args"];
      if (call_args.size() != 1)
        throw std::runtime_error("type() expects exactly 1 argument");

      const auto &type_call_arg = call_args[0];

      // Handle type(None)
      if (type_call_arg["_type"] == "Constant")
      {
        if (
          type_call_arg["value"].is_null() ||
          (type_call_arg.contains("value") &&
           type_call_arg["value"] == nullptr))
        {
          return "NoneType";
        }
        throw std::runtime_error(
          "isinstance() with type(constant) only supports type(None)");
      }
      else if (type_call_arg["_type"] == "Name")
      {
        throw std::runtime_error(
          "isinstance() with type(variable) not yet supported - use direct "
          "type names");
      }
      else
        throw std::runtime_error(
          "Unsupported argument to type() in isinstance()");
    }
    else
      throw std::runtime_error("Unsupported type format in isinstance()");
  };

  // Build isinstance check for a given type name
  auto build_isinstance = [&](const std::string &type_name) -> exprt {
    // Special case: Check if object is None (null pointer)
    if (type_name == "NoneType")
    {
      // If x is not a pointer type, it can never be None
      if (!obj_expr.type().is_pointer())
        return gen_zero(typet("bool")); // false

      // For pointer types, check if it's null
      exprt null_ptr = gen_zero(obj_expr.type());
      exprt equality("=", typet("bool"));
      equality.copy_to_operands(obj_expr);
      equality.move_to_operands(null_ptr);
      return equality;
    }

    // Regular type checking
    typet expected_type = type_handler_.get_typet(type_name, 0);
    if (expected_type.is_nil())
      throw std::runtime_error("Could not resolve type: " + type_name);

    exprt t;

    if (expected_type.is_pointer())
    {
      const pointer_typet &ptr_type = to_pointer_type(expected_type);
      const typet &pointee_type = ptr_type.subtype();

      if (pointee_type.is_symbol())
      {
        const symbolt *symbol = converter_.ns.lookup(pointee_type);
        if (!symbol)
          throw std::runtime_error(
            "Could not find symbol for type: " + type_name);
        t = symbol_expr(*symbol);
      }
      else
        t = gen_zero(pointee_type);
    }
    else if (expected_type.is_symbol())
    {
      const symbolt *symbol = converter_.ns.lookup(expected_type);
      if (!symbol)
        throw std::runtime_error(
          "Could not find symbol for type: " + type_name);
      t = symbol_expr(*symbol);
    }
    else
      t = gen_zero(expected_type);

    exprt isinstance("isinstance", typet("bool"));
    isinstance.copy_to_operands(obj_expr);
    isinstance.move_to_operands(t);
    return isinstance;
  };

  // Handle tuple of types: isinstance(v, (int, str, type(None)))
  if (type_arg["_type"] == "Tuple")
  {
    const auto &elts = type_arg["elts"];
    if (elts.empty())
      throw std::runtime_error("isinstance() tuple cannot be empty");

    // Build isinstance check for first element
    exprt result = build_isinstance(extract_type_name(elts[0]));

    // Chain OR expressions for remaining elements
    for (size_t i = 1; i < elts.size(); ++i)
    {
      exprt next_check = build_isinstance(extract_type_name(elts[i]));

      exprt or_expr("or", typet("bool"));
      or_expr.move_to_operands(result);
      or_expr.move_to_operands(next_check);
      result = or_expr;
    }

    return result;
  }

  // Handle single type: isinstance(v, int) or isinstance(v, type(None))
  return build_isinstance(extract_type_name(type_arg));
}

exprt function_call_expr::handle_hasattr() const
{
  const auto &args = call_["args"];
  if (args.size() != 2)
    throw std::runtime_error("hasattr() takes exactly 2 arguments");

  const exprt &obj_expr = converter_.get_expr(args[0]);
  const auto &attr_arg = args[1];

  if (
    attr_arg["_type"] != "Constant" || !attr_arg.contains("value") ||
    !attr_arg["value"].is_string())
    throw std::runtime_error(
      "hasattr() expects attribute name as string literal");

  std::string attr_name = attr_arg["value"].get<std::string>();
  typet attr_type = array_typet(
    unsigned_char_type(), from_integer(attr_name.size() + 1, size_type()));
  string_constantt attr_expr(attr_name, attr_type, string_constantt::k_default);

  exprt hasattr("hasattr", typet("bool"));
  hasattr.copy_to_operands(obj_expr);
  hasattr.move_to_operands(attr_expr);
  return hasattr;
}

exprt function_call_expr::handle_divmod() const
{
  const auto &args = call_["args"];

  if (args.size() != 2)
    throw std::runtime_error("divmod() takes exactly 2 arguments");

  exprt dividend = converter_.get_expr(args[0]);
  exprt divisor = converter_.get_expr(args[1]);

  return converter_.get_math_handler().handle_divmod(dividend, divisor, call_);
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
  bool is_constant = false;

  // Check for unary minus: e.g., chr(-1)
  if (arg.contains("_type") && arg["_type"] == "UnaryOp")
  {
    const auto &op = arg["op"];
    const auto &operand = arg["operand"];

    if (
      op["_type"] == "USub" && operand.contains("value") &&
      operand["value"].is_number_integer())
    {
      int_value = -operand["value"].get<int>();
      is_constant = true;
    }
    else
      return converter_.get_exception_handler().gen_exception_raise(
        "TypeError", "Unsupported UnaryOp in chr()");
  }

  // Handle integer input
  else if (arg.contains("value") && arg["value"].is_number_integer())
  {
    int_value = arg["value"].get<int>();
    is_constant = true;
  }

  // Reject float input
  else if (arg.contains("value") && arg["value"].is_number_float())
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError", "chr() argument must be int, not float");

  // Try converting string input to integer
  else if (arg.contains("value") && arg["value"].is_string())
  {
    const std::string &s = arg["value"].get<std::string>();
    try
    {
      int_value = std::stoi(s);
      is_constant = true;
    }
    catch (const std::invalid_argument &)
    {
      return converter_.get_exception_handler().gen_exception_raise(
        "TypeError", "invalid string passed to chr()");
    }
  }

  // Handle variable references (Name nodes)
  else if (arg.contains("_type") && arg["_type"] == "Name")
  {
    const symbolt *sym = lookup_python_symbol(arg["id"]);
    if (!sym)
    {
      // Symbol not found - use runtime conversion via string_handler
      exprt var_expr = converter_.get_expr(arg);
      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_chr_conversion(
        var_expr, loc);
    }

    exprt val = sym->value;

    if (!val.is_constant())
      val = converter_.get_resolved_value(val);

    if (val.is_nil())
    {
      // Runtime variable: use string_handler for runtime conversion
      exprt var_expr = converter_.get_expr(arg);
      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_chr_conversion(
        var_expr, loc);
    }

    // Even if the symbol has a constant value, we should not
    // perform compile-time conversion if the symbol is mutable (lvalue)
    // because its value may change at runtime (e.g., loop variables)
    if (sym->lvalue)
    {
      // This is a mutable variable - use runtime conversion
      exprt var_expr = converter_.get_expr(arg);
      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_chr_conversion(
        var_expr, loc);
    }

    // Try to extract constant value (only for truly constant symbols)
    const auto &const_expr = to_constant_expr(val);
    std::string binary_str = id2string(const_expr.get_value());
    try
    {
      int_value = std::stoul(binary_str, nullptr, 2);
      is_constant = true;
    }
    catch (std::out_of_range &)
    {
      return converter_.get_exception_handler().gen_exception_raise(
        "ValueError", "chr() argument outside of Unicode range");
    }
    catch (std::invalid_argument &)
    {
      return converter_.get_exception_handler().gen_exception_raise(
        "TypeError", "must be of type int");
    }

    arg["_type"] = "Constant";
    arg.erase("id");
    arg.erase("ctx");
  }

  // If we have a constant value, do compile-time conversion
  if (is_constant)
  {
    std::string utf8_encoded;

    try
    {
      utf8_encoded = utf8_encode(int_value);
    }
    catch (const std::out_of_range &e)
    {
      return converter_.get_exception_handler().gen_exception_raise(
        "ValueError", "chr()");
    }

    // Build a proper character array, not a single char
    // Create array type with proper size (length + null terminator)
    size_t array_size = utf8_encoded.size() + 1;
    typet char_array_type =
      array_typet(char_type(), from_integer(array_size, size_type()));

    // Build the array expression with all characters
    std::vector<unsigned char> char_data(
      utf8_encoded.begin(), utf8_encoded.end());
    char_data.push_back('\0'); // Add null terminator

    // Use converter's make_char_array_expr to build proper array
    exprt result = converter_.make_char_array_expr(char_data, char_array_type);

    return result;
  }

  // If we reach here, we need runtime conversion
  exprt var_expr = converter_.get_expr(arg);
  locationt loc = converter_.get_location_from_decl(call_);
  return converter_.get_string_handler().handle_chr_conversion(var_expr, loc);
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
      return converter_.get_exception_handler().gen_exception_raise(
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
    return converter_.get_exception_handler().gen_exception_raise(
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
      return converter_.get_exception_handler().gen_exception_raise(
        "NameError", "variable '" + var_name + "' is not defined");
    }

    typet operand_type = sym->value.type();
    std::string py_type = type_handler_.type_to_string(operand_type);

    if (operand_type != char_type() && py_type != "str")
    {
      return converter_.get_exception_handler().gen_exception_raise(
        "TypeError",
        "ord() expected string of length 1, but " + py_type + " found");
    }

    // For runtime variables (mutable), try to extract constant value if available
    if (sym->lvalue && !sym->value.is_nil())
    {
      auto value_opt = extract_string_from_symbol(sym);
      if (value_opt)
      {
        // Successfully extracted constant value from lvalue string
        code_point = decode_utf8_codepoint(*value_opt);

        // Remove Name data
        arg["_type"] = "Constant";
        arg.erase("id");
        arg.erase("ctx");

        // Replace the arg with the integer value
        arg["value"] = code_point;
        arg["type"] = "int";

        // Build and return the integer expression
        exprt expr = converter_.get_expr(arg);
        expr.type() = type_handler_.get_typet("int", 0);
        return expr;
      }
      // If extraction failed for lvalue, fall through to runtime conversion
    }

    // Use runtime conversion for variables without constant value or failed extraction
    if (sym->value.is_nil() || sym->lvalue)
    {
      exprt var_expr = converter_.get_expr(arg);

      if (
        var_expr.type() == char_type() || var_expr.type().is_signedbv() ||
        var_expr.type().is_unsignedbv())
      {
        return typecast_exprt(var_expr, int_type());
      }

      return converter_.get_exception_handler().gen_exception_raise(
        "ValueError", "ord() requires a character");
    }

    // Compile-time extraction for constant symbols
    auto value_opt = extract_string_from_symbol(sym);
    if (!value_opt)
    {
      return converter_.get_exception_handler().gen_exception_raise(
        "ValueError", "failed to extract string from symbol");
    }

    code_point = decode_utf8_codepoint(*value_opt);

    // Remove Name data
    arg["_type"] = "Constant";
    arg.erase("id");
    arg.erase("ctx");
  }
  else
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError", "ord() argument must be a string");

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
  std::string enclosing_function = converter_.current_function_name();

  // Construct the full symbol identifier with function scope
  std::string var_symbol =
    "py:" + filename + "@F@" + enclosing_function + "@" + var_name;
  const symbolt *sym = converter_.find_symbol(var_symbol);

  // If not found in function scope, try module-level scope
  if (!sym)
  {
    var_symbol = "py:" + filename + "@" + var_name;
    sym = converter_.find_symbol(var_symbol);
  }

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
    return converter_.get_exception_handler().gen_exception_raise(
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

      exprt dunder_result = converter_.dispatch_unary_dunder_operator(
        "abs", inferred_expr, converter_.get_location_from_decl(call_));
      if (!dunder_result.is_nil())
        return dunder_result;

      // Build a symbolic abs() expression with the resolved operand type
      exprt abs_expr("abs", inferred_type);
      abs_expr.copy_to_operands(inferred_expr);
      return abs_expr;
    }
    catch (const std::exception &e)
    {
      return converter_.get_exception_handler().gen_exception_raise(
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
      exprt operand_expr = converter_.get_expr(arg);
      typet operand_type = operand_expr.type();

      exprt dunder_result = converter_.dispatch_unary_dunder_operator(
        "abs", operand_expr, converter_.get_location_from_decl(call_));
      if (!dunder_result.is_nil())
        return dunder_result;

      // Build a symbolic abs() expression with the resolved operand type
      exprt abs_expr("abs", operand_type);
      abs_expr.copy_to_operands(operand_expr);

      return abs_expr;
    }
    else
    {
      // Variable could not be resolved
      return converter_.get_exception_handler().gen_exception_raise(
        "NameError", "variable '" + var_name + "' is not defined");
    }
  }

  // Final fallback if no type is available
  std::string arg_type = arg.value("type", "");
  if (arg_type.empty())
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError", "operand to abs() is missing a type");

  // Only numeric types are valid operands for abs()
  if (arg_type != "int" && arg_type != "float" && arg_type != "complex")
    return converter_.get_exception_handler().gen_exception_raise(
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

  // Handle int(): convert to integer
  else if (func_name == "int")
  {
    // Get the arguments list from the call
    const nlohmann::json &arguments =
      call_.contains("args") ? call_["args"] : nlohmann::json::array();

    // int() with no arguments returns 0
    if (arguments.empty())
    {
      return from_integer(0, int_type());
    }

    const nlohmann::json &first_arg = arguments[0];

    // Check if we have a base argument (second parameter)
    exprt base_expr = nil_exprt();
    if (arguments.size() > 1)
    {
      // Get the base expression
      base_expr = converter_.get_expr(arguments[1]);
    }

    // Handle Name type (variable reference)
    if (first_arg["_type"] == "Name")
    {
      const symbolt *sym = lookup_python_symbol(first_arg["id"]);
      if (sym && sym->value.is_constant())
      {
        if (base_expr.is_nil())
        {
          return handle_str_symbol_to_int(sym);
        }
        else
        {
          // Convert symbol to expression and use with base
          exprt value_expr = symbol_expr(*sym);
          return converter_.get_string_handler()
            .handle_int_conversion_with_base(
              value_expr, base_expr, converter_.get_location_from_decl(call_));
        }
      }
      else
      {
        // Try to get the expression type directly
        exprt expr = converter_.get_expr(first_arg);

        if (base_expr.is_nil())
        {
          // No base provided, use general conversion
          return converter_.get_string_handler().handle_int_conversion(
            expr, converter_.get_location_from_decl(call_));
        }
        else
        {
          // Base provided, use conversion with base
          return converter_.get_string_handler()
            .handle_int_conversion_with_base(
              expr, base_expr, converter_.get_location_from_decl(call_));
        }
      }
    }
    // Handle other types (Constant, etc.)
    else
    {
      exprt value_expr = converter_.get_expr(first_arg);

      // If it's a constant string, we need to ensure proper conversion
      if (
        first_arg["_type"] == "Constant" && first_arg.contains("value") &&
        first_arg["value"].is_string())
      {
        // This is a string literal - use string conversion
        if (base_expr.is_nil())
        {
          return converter_.get_string_handler().handle_string_to_int_base10(
            value_expr, converter_.get_location_from_decl(call_));
        }
        else
        {
          return converter_.get_string_handler().handle_string_to_int(
            value_expr, base_expr, converter_.get_location_from_decl(call_));
        }
      }

      if (base_expr.is_nil())
      {
        // No base provided
        return converter_.get_string_handler().handle_int_conversion(
          value_expr, converter_.get_location_from_decl(call_));
      }
      else
      {
        // Base provided
        return converter_.get_string_handler().handle_int_conversion_with_base(
          value_expr, base_expr, converter_.get_location_from_decl(call_));
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
        return converter_.get_exception_handler().gen_exception_raise(
          "ValueError", m);
      }
      catch (const std::out_of_range &)
      {
        std::string m = "could not convert string to float : '" +
                        arg["value"].get<std::string>() + "' (out of range)";
        return converter_.get_exception_handler().gen_exception_raise(
          "ValueError", m);
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

        return converter_.get_exception_handler().gen_exception_raise(
          "ValueError", m);
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
  bool is_min_or_max = (func_name == "min" || func_name == "max");

  if (!is_min_or_max)
    return false;

  const auto &args = call_["args"];

  // Handle two-argument case: min(a, b) or max(a, b)
  if (args.size() == 2)
    return true;

  // Handle single-argument case if it's a tuple
  if (args.size() == 1)
  {
    exprt arg = converter_.get_expr(args[0]);
    const typet &arg_type = converter_.ns.follow(arg.type());

    // Check if it's a tuple (struct with tag-tuple prefix)
    if (arg_type.id() == "struct")
    {
      const struct_typet &struct_type = to_struct_type(arg_type);
      std::string tag = struct_type.tag().as_string();
      return tag.starts_with("tag-tuple");
    }
  }

  // Single argument that's not a tuple falls through to general handler (for lists)
  return false;
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
  {
    // Single iterable argument case: min(iterable) or max(iterable)
    exprt arg = converter_.get_expr(args[0]);
    const typet &arg_type = converter_.ns.follow(arg.type());

    // Check if it's a tuple (struct type with element_N components)
    if (arg_type.is_struct())
    {
      const struct_typet &struct_type = to_struct_type(arg_type);

      // Check if this is a tuple by examining the tag
      std::string tag = struct_type.tag().as_string();
      if (tag.starts_with("tag-tuple"))
      {
        // Handle tuple directly by building comparison chain
        const auto &components = struct_type.components();

        if (components.empty())
          throw std::runtime_error(func_name + "() arg is an empty sequence");

        // Start with first element: result = t.element_0
        exprt result =
          member_exprt(arg, components[0].get_name(), components[0].type());

        // Compare with remaining elements
        for (size_t i = 1; i < components.size(); ++i)
        {
          member_exprt elem(
            arg, components[i].get_name(), components[i].type());

          // Create comparison: elem < result (for min) or elem > result (for max)
          exprt condition(comparison_op, type_handler_.get_typet("bool", 0));
          condition.copy_to_operands(elem, result);

          // result = (elem < result) ? elem : result
          if_exprt update(condition, elem, result);
          update.type() = components[i].type();
          result = update;
        }

        return result;
      }
    }
  }

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
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_clear");
  assert(clear_func);

  // Build function call
  code_function_callt clear_call;
  clear_call.function() = symbol_expr(*clear_func);
  clear_call.arguments().push_back(symbol_expr(*list_symbol));
  clear_call.type() = empty_typet();
  clear_call.location() = converter_.get_location_from_decl(call_);

  return clear_call;
}

exprt function_call_expr::handle_list_pop() const
{
  const auto &args = call_["args"];

  // pop() can take 0 or 1 arguments (default pops last element)
  if (args.size() > 1)
    throw std::runtime_error("pop() takes at most 1 argument");

  // Get the list object name
  std::string list_name = get_object_name();

  // Find the list symbol
  symbol_id list_symbol_id = converter_.create_symbol_id();
  list_symbol_id.set_object(list_name);
  const symbolt *list_symbol =
    converter_.find_symbol(list_symbol_id.to_string());

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_name);

  // Determine the index (default is -1 for last element)
  exprt index_expr;
  if (args.empty())
  {
    // No argument: pop last element (index -1)
    index_expr = from_integer(-1, signedbv_typet(64));
  }
  else
  {
    // Use provided index
    index_expr = converter_.get_expr(args[0]);
  }

  // Delegate to python_list to build the pop operation
  python_list list_helper(converter_, call_);
  return list_helper.build_pop_list_call(*list_symbol, index_expr, call_);
}

bool function_call_expr::is_dict_method_call() const
{
  if (call_["func"]["_type"] != "Attribute")
    return false;

  const std::string &method_name = function_id_.get_function();

  // Check if this is a known dict method
  return method_name == "get";
}

exprt function_call_expr::handle_dict_method() const
{
  const std::string &method_name = function_id_.get_function();

  if (method_name == "get")
  {
    // Get the dict object
    std::string dict_name = get_object_name();

    symbol_id dict_symbol_id = converter_.create_symbol_id();
    dict_symbol_id.set_object(dict_name);
    const symbolt *dict_symbol =
      converter_.find_symbol(dict_symbol_id.to_string());

    if (!dict_symbol)
      throw std::runtime_error("Dictionary variable not found: " + dict_name);

    // Delegate to dict handler
    return converter_.get_dict_handler()->handle_dict_get(
      symbol_expr(*dict_symbol), call_);
  }

  throw std::runtime_error("Unsupported dict method: " + method_name);
}

exprt function_call_expr::handle_list_copy() const
{
  const auto &args = call_["args"];

  if (!args.empty())
    throw std::runtime_error("copy() takes no arguments");

  // Get the list object name
  std::string list_name = get_object_name();

  // Find the list symbol
  symbol_id list_symbol_id = converter_.create_symbol_id();
  list_symbol_id.set_object(list_name);
  const symbolt *list_symbol =
    converter_.find_symbol(list_symbol_id.to_string());

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_name);

  // Delegate to python_list to build the copy operation
  python_list list_helper(converter_, call_);
  return list_helper.build_copy_list_call(*list_symbol, call_);
}

exprt function_call_expr::handle_list_remove() const
{
  const auto &args = call_["args"];

  if (args.size() != 1)
    throw std::runtime_error("remove() takes exactly one argument");

  std::string list_name = get_object_name();

  symbol_id list_symbol_id = converter_.create_symbol_id();
  list_symbol_id.set_object(list_name);
  const symbolt *list_symbol =
    converter_.find_symbol(list_symbol_id.to_string());

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_name);

  exprt value_to_remove = converter_.get_expr(args[0]);

  python_list list_helper(converter_, call_);
  exprt result =
    list_helper.build_remove_list_call(*list_symbol, call_, value_to_remove);

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
         method_name == "clear" || method_name == "extend" ||
         method_name == "copy";
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
  if (method_name == "pop")
    return handle_list_pop();
  if (method_name == "copy")
    return handle_list_copy();
  if (method_name == "remove")
    return handle_list_remove();

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

  // If value_to_append is a function call, materialize its return value
  bool is_func_call = (value_to_append.is_code() &&
                       value_to_append.get("statement") == "function_call") ||
                      (value_to_append.id() == "sideeffect" &&
                       value_to_append.get("statement") == "function_call");

  if (is_func_call)
  {
    exprt func_expr;
    exprt::operandst func_args;
    typet ret_type;

    if (value_to_append.is_code())
    {
      const code_function_callt &call =
        to_code_function_call(to_code(value_to_append));
      func_expr = call.function();
      func_args = call.arguments();
      ret_type = call.type();
    }
    else
    {
      // side_effect_expr_function_callt
      const side_effect_expr_function_callt &call =
        to_side_effect_expr_function_call(value_to_append);
      func_expr = call.function();
      func_args = call.arguments();
      ret_type = call.type();
    }

    if (ret_type.is_nil() || ret_type.is_empty())
    {
      log_warning(
        "list.append with function call: unknown return type, assuming int");
      ret_type = int_type();
    }

    symbolt &tmp_var = converter_.create_tmp_symbol(
      call_, "$append_ret$", ret_type, gen_zero(ret_type));

    code_declt tmp_decl(symbol_expr(tmp_var));
    tmp_decl.location() = converter_.get_location_from_decl(call_);
    converter_.current_block->copy_to_operands(tmp_decl);

    // Create function call with lhs
    code_function_callt new_call;
    new_call.function() = func_expr;
    new_call.arguments() = func_args;
    new_call.lhs() = symbol_expr(tmp_var);
    new_call.type() = ret_type;
    new_call.location() = converter_.get_location_from_decl(call_);
    converter_.current_block->copy_to_operands(new_call);

    // Replace value_to_append with the temporary variable
    value_to_append = symbol_expr(tmp_var);
  }

  if (
    value_to_append.type().is_array() &&
    value_to_append.type().subtype() == char_type())
  {
    const array_typet &array_type = to_array_type(value_to_append.type());
    // Only convert single-element char arrays (string literals)
    if (array_type.size().is_constant())
    {
      const constant_exprt &size_const = to_constant_expr(array_type.size());
      BigInt size_value = binary2integer(size_const.value().c_str(), false);
      if (size_value == 1)
        value_to_append.type() = gen_pointer_type(char_type());
    }
  }

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
      return converter_.get_exception_handler().gen_exception_raise(
        "TypeError", msg.str());
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

bool function_call_expr::is_math_comb_call() const
{
  const std::string &func_name = function_id_.get_function();

  // Check if it's the wrapper function
  if (func_name == "comb")
  {
    // Verify it's being called from the math module
    if (call_["func"]["_type"] == "Attribute")
    {
      std::string caller = get_object_name();
      return (caller == "math");
    }
  }

  return false;
}

exprt function_call_expr::handle_math_comb() const
{
  const auto &args = call_["args"];

  if (args.size() != 2)
    throw std::runtime_error("comb() takes exactly 2 arguments");

  // Get the argument expressions
  exprt n_expr = converter_.get_expr(args[0]);
  exprt k_expr = converter_.get_expr(args[1]);

  // Type checking: both arguments must be integers
  if (!n_expr.type().is_signedbv() && !n_expr.type().is_unsignedbv())
  {
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError",
      "'" + type_handler_.type_to_string(n_expr.type()) +
        "' object cannot be interpreted as an integer");
  }

  if (!k_expr.type().is_signedbv() && !k_expr.type().is_unsignedbv())
  {
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError",
      "'" + type_handler_.type_to_string(k_expr.type()) +
        "' object cannot be interpreted as an integer");
  }

  // Find the actual comb implementation function
  const symbolt *comb_func = converter_.find_symbol(function_id_.to_string());

  if (!comb_func)
    throw std::runtime_error("comb() implementation not found");

  // Build the function call
  locationt location = converter_.get_location_from_decl(call_);
  code_function_callt call;
  call.location() = location;
  call.function() = symbol_expr(*comb_func);
  call.type() = int_type();
  call.arguments().push_back(n_expr);
  call.arguments().push_back(k_expr);

  return call;
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

    // Dict methods
    {[this]() { return is_dict_method_call(); },
     [this]() { return handle_dict_method(); },
     "dict methods"},

    // Math module functions (isnan, isinf)
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

    // Math module functions (sin, cos, sqrt, exp, log, etc.)
    {[this]() {
       const std::string &func_name = function_id_.get_function();
       bool is_math_module = false;
       if (call_["func"]["_type"] == "Attribute")
       {
         std::string caller = get_object_name();
         is_math_module = (caller == "math");
       }

       bool is_math_wrapper =
         (func_name == "__ESBMC_sin" || func_name == "__ESBMC_cos" ||
          func_name == "__ESBMC_sqrt" || func_name == "__ESBMC_exp" ||
          func_name == "__ESBMC_log" || func_name == "__ESBMC_acos" ||
          func_name == "__ESBMC_atan" || func_name == "__ESBMC_atan2" ||
          func_name == "__ESBMC_log2" || func_name == "__ESBMC_pow" ||
          func_name == "__ESBMC_fabs" || func_name == "__ESBMC_trunc" ||
          func_name == "__ESBMC_fmod" || func_name == "__ESBMC_copysign" ||
          func_name == "__ESBMC_tan" || func_name == "__ESBMC_asin" ||
          func_name == "__ESBMC_sinh" || func_name == "__ESBMC_cosh" ||
          func_name == "__ESBMC_tanh" || func_name == "__ESBMC_log10" ||
          func_name == "__ESBMC_expm1" || func_name == "__ESBMC_log1p" ||
          func_name == "__ESBMC_exp2" || func_name == "__ESBMC_asinh" ||
          func_name == "__ESBMC_acosh" || func_name == "__ESBMC_atanh" ||
          func_name == "__ESBMC_hypot");

       return (is_math_module &&
               (func_name == "sin" || func_name == "cos" ||
                func_name == "sqrt" || func_name == "exp" ||
                func_name == "log" || func_name == "acos" ||
                func_name == "atan" || func_name == "atan2" ||
                func_name == "log2" || func_name == "pow" ||
                func_name == "fabs" || func_name == "trunc" ||
                func_name == "fmod" || func_name == "copysign" ||
                func_name == "tan" || func_name == "asin" ||
                func_name == "sinh" || func_name == "cosh" ||
                func_name == "tanh" || func_name == "log10" ||
                func_name == "expm1" || func_name == "log1p" ||
                func_name == "exp2" || func_name == "asinh" ||
                func_name == "acosh" || func_name == "atanh" ||
                func_name == "hypot" || func_name == "cbrt" ||
                func_name == "erf" || func_name == "erfc" ||
                func_name == "frexp" || func_name == "fsum" ||
                func_name == "gamma" || func_name == "ldexp" ||
                func_name == "lgamma" || func_name == "nextafter" ||
                func_name == "remainder" || func_name == "sumprod" ||
                func_name == "ulp")) ||
              is_math_wrapper;
     },
     [this]() -> exprt {
       const std::string &func_name = function_id_.get_function();
       const auto &args = call_["args"];

       auto require_one_arg = [&]() -> exprt {
         if (args.size() != 1)
           throw std::runtime_error(
             func_name + "() expects exactly 1 argument");
         return converter_.get_expr(args[0]);
       };

       auto require_two_args = [&]() -> std::pair<exprt, exprt> {
         if (args.size() != 2)
           throw std::runtime_error(
             func_name + "() expects exactly 2 arguments");
         return {converter_.get_expr(args[0]), converter_.get_expr(args[1])};
       };

       if (func_name == "sin" || func_name == "__ESBMC_sin")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_sin(arg_expr, call_);
       }
       else if (func_name == "cos" || func_name == "__ESBMC_cos")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_cos(arg_expr, call_);
       }
       else if (func_name == "exp" || func_name == "__ESBMC_exp")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_exp(arg_expr, call_);
       }
       else if (func_name == "sqrt" || func_name == "__ESBMC_sqrt")
       {
         exprt arg_expr = require_one_arg();
         // Domain check for sqrt: operand must be >= 0
         exprt double_operand = arg_expr;
         if (!arg_expr.type().is_floatbv())
         {
           double_operand =
             exprt("typecast", type_handler_.get_typet("float", 0));
           double_operand.copy_to_operands(arg_expr);
         }

         exprt zero = gen_zero(type_handler_.get_typet("float", 0));
         exprt domain_check = exprt("<", type_handler_.get_typet("bool", 0));
         domain_check.copy_to_operands(double_operand, zero);

         // Create the exception raise as a code expression
         exprt raise_expr =
           converter_.get_exception_handler().gen_exception_raise(
             "ValueError", "math domain error");
         locationt loc = converter_.get_location_from_decl(call_);
         raise_expr.location() = loc;
         raise_expr.location().user_provided(true);

         // Convert expression to code statement
         code_expressiont raise_code(raise_expr);
         raise_code.location() = loc;

         // Create the guard condition
         code_ifthenelset guard;
         guard.cond() = domain_check;
         guard.then_case() = raise_code;
         guard.location() = loc;

         // Add the guard to the current block
         converter_.current_block->copy_to_operands(guard);

         // Now compute sqrt (only reached if operand >= 0)
         exprt sqrt_result =
           converter_.get_math_handler().handle_sqrt(arg_expr, call_);

         return sqrt_result;
       }
       else if (func_name == "log" || func_name == "__ESBMC_log")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_log(arg_expr, call_);
       }
       else if (func_name == "acos" || func_name == "__ESBMC_acos")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_acos(arg_expr, call_);
       }
       else if (func_name == "atan" || func_name == "__ESBMC_atan")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_atan(arg_expr, call_);
       }
       else if (func_name == "log2" || func_name == "__ESBMC_log2")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_log2(arg_expr, call_);
       }
       else if (func_name == "tan" || func_name == "__ESBMC_tan")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_tan(arg_expr, call_);
       }
       else if (func_name == "asin" || func_name == "__ESBMC_asin")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_asin(arg_expr, call_);
       }
       else if (func_name == "sinh" || func_name == "__ESBMC_sinh")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_sinh(arg_expr, call_);
       }
       else if (func_name == "cosh" || func_name == "__ESBMC_cosh")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_cosh(arg_expr, call_);
       }
       else if (func_name == "tanh" || func_name == "__ESBMC_tanh")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_tanh(arg_expr, call_);
       }
       else if (func_name == "log10" || func_name == "__ESBMC_log10")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_log10(arg_expr, call_);
       }
       else if (func_name == "expm1" || func_name == "__ESBMC_expm1")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_expm1(arg_expr, call_);
       }
       else if (func_name == "log1p" || func_name == "__ESBMC_log1p")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_log1p(arg_expr, call_);
       }
       else if (func_name == "exp2" || func_name == "__ESBMC_exp2")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_exp2(arg_expr, call_);
       }
       else if (func_name == "asinh" || func_name == "__ESBMC_asinh")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_asinh(arg_expr, call_);
       }
       else if (func_name == "acosh" || func_name == "__ESBMC_acosh")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_acosh(arg_expr, call_);
       }
       else if (func_name == "atanh" || func_name == "__ESBMC_atanh")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_atanh(arg_expr, call_);
       }
       else if (func_name == "fabs" || func_name == "__ESBMC_fabs")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_fabs(arg_expr, call_);
       }
       else if (func_name == "trunc" || func_name == "__ESBMC_trunc")
       {
         exprt arg_expr = require_one_arg();
         return converter_.get_math_handler().handle_trunc(arg_expr, call_);
       }
       else if (func_name == "atan2" || func_name == "__ESBMC_atan2")
       {
         auto [y_expr, x_expr] = require_two_args();
         return converter_.get_math_handler().handle_atan2(
           y_expr, x_expr, call_);
       }
       else if (func_name == "pow" || func_name == "__ESBMC_pow")
       {
         auto [base_expr, exp_expr] = require_two_args();
         return converter_.get_math_handler().handle_pow(
           base_expr, exp_expr, call_);
       }
       else if (func_name == "fmod" || func_name == "__ESBMC_fmod")
       {
         auto [lhs_expr, rhs_expr] = require_two_args();
         return converter_.get_math_handler().handle_fmod(
           lhs_expr, rhs_expr, call_);
       }
       else if (func_name == "copysign" || func_name == "__ESBMC_copysign")
       {
         auto [lhs_expr, rhs_expr] = require_two_args();
         return converter_.get_math_handler().handle_copysign(
           lhs_expr, rhs_expr, call_);
       }
       else if (func_name == "hypot" || func_name == "__ESBMC_hypot")
       {
         auto [lhs_expr, rhs_expr] = require_two_args();
         return converter_.get_math_handler().handle_hypot(
           lhs_expr, rhs_expr, call_);
       }
       else if (
         func_name == "cbrt" || func_name == "erf" || func_name == "erfc" ||
         func_name == "frexp" || func_name == "fsum" || func_name == "gamma" ||
         func_name == "ldexp" || func_name == "lgamma" ||
         func_name == "nextafter" || func_name == "remainder" ||
         func_name == "sumprod" || func_name == "ulp")
       {
         return handle_general_function_call();
       }

       throw std::runtime_error("Unsupported math function: " + func_name);
     },
     "math.sin/cos/sqrt/exp/log/etc"},

    // Math.comb function with type checking
    {[this]() { return is_math_comb_call(); },
     [this]() { return handle_math_comb(); },
     "math.comb"},

    // divmod function
    {[this]() {
       const std::string &func_name = function_id_.get_function();
       return func_name == "divmod";
     },
     [this]() { return handle_divmod(); },
     "divmod"},

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

  // Handle single-argument min/max by dispatching to typed builtins
  const std::string &func_name = function_id_.get_function();
  std::string actual_func_name = func_name;

  if (
    (func_name == "min" || func_name == "max" || func_name == "sorted") &&
    call_["args"].size() == 1)
  {
    exprt list_arg = converter_.get_expr(call_["args"][0]);
    typet elem_type;
    if (list_arg.is_symbol())
    {
      const std::string &list_id = list_arg.identifier().as_string();
      // Check that all elements have the same type and get the common type
      elem_type = python_list::check_homogeneous_list_types(list_id, func_name);
    }
    // Dispatch to typed builtin based on element type
    if (!elem_type.is_nil())
    {
      if (elem_type.is_floatbv())
        actual_func_name += "_float";
      else if (
        (elem_type.is_pointer() && elem_type.subtype() == char_type()) ||
        (elem_type.is_array() && elem_type.subtype() == char_type()))
        actual_func_name += "_str";
      // Integer types use base name without suffix
    }
  }

  // Get object symbol
  symbolt *obj_symbol = nullptr;
  symbol_id obj_symbol_id = converter_.create_symbol_id();

  if (call_["func"]["_type"] == "Attribute")
  {
    std::string caller = get_object_name();
    obj_symbol_id.set_object(caller);
    obj_symbol = symbol_table.find_symbol(obj_symbol_id.to_string());
  }

  // Get function symbol id - use actual_func_name for typed dispatch
  std::string func_symbol_id;
  if (actual_func_name != func_name)
  {
    symbol_id modified_function_id = function_id_;
    modified_function_id.set_function(actual_func_name);
    func_symbol_id = modified_function_id.to_string();
  }
  else
    func_symbol_id = function_id_.to_string();

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
          // Create a side effect expression with nondet for assignments
          locationt location = converter_.get_location_from_decl(call_);
          // Create a nondet expression as a placeholder that won't crash
          // This allows the code to continue but marks it as undefined behavior
          exprt nondet_expr("sideeffect", empty_typet());
          nondet_expr.statement("nondet");
          nondet_expr.location() = location;
          nondet_expr.location().user_provided(true);
          nondet_expr.location().comment(
            "Unsupported function '" + func_name + "' called");
          // Also add an assertion to the current block to flag this as an error
          exprt false_expr = gen_boolean(false);
          code_assertt assert_code(false_expr);
          assert_code.location() = location;
          assert_code.location().user_provided(true);
          assert_code.location().comment(
            "Unsupported function '" + func_name + "' is reached");
          converter_.current_block->copy_to_operands(assert_code);

          return nondet_expr;
        }
      }
    }
  }

  if (func_symbol != nullptr)
  {
    exprt type_error =
      check_argument_types(func_symbol, call_["args"], call_["keywords"]);
    if (!type_error.is_nil())
      return type_error;
  }

  locationt location = converter_.get_location_from_decl(call_);

  code_function_callt call;
  call.location() = location;
  call.function() = symbol_expr(*func_symbol);
  const typet &return_type = to_code_type(func_symbol->type).return_type();
  call.type() = return_type;

  // Determine parameter offset for Optional wrapping logic
  size_t param_offset = 0;

  // Add self as first parameter
  if (function_type_ == FunctionType::Constructor)
  {
    call.type() = type_handler_.get_typet(func_symbol->name.as_string());

    // Self is the LHS
    if (converter_.current_lhs)
    {
      call.arguments().push_back(gen_address_of(*converter_.current_lhs));
      param_offset = 1;
    }
    else
    {
      // For constructor calls without assignment, delay creating temp var
      // get_return_statements() will handle return statements, we only handle
      // standalone calls (e.g., Positive(2))
      // Self parameter will be added later if needed (see end of function)
      // param_offset is 1 because first user arg maps to param[1] (skipping self)
      param_offset = 1;
    }
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
    param_offset = 1;
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
      param_offset = 1;
    }
    else
    {
      // Passing a void pointer to the "cls" argument
      typet t = pointer_typet(empty_typet());
      call.arguments().push_back(gen_zero(t));
      param_offset = 1;

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

  // Get function type and parameters for Optional wrapping
  const code_typet &func_type = to_code_type(func_symbol->type);
  const auto &params = func_type.arguments();

  size_t arg_index = 0;
  for (const auto &arg_node : call_["args"])
  {
    exprt arg = converter_.get_expr(arg_node);

    // Check if the corresponding parameter is Optional
    size_t param_idx = arg_index + param_offset;

    if (param_idx < params.size())
    {
      const typet &param_type = params[param_idx].type();

      // Handle character-to-string-pointer conversion
      if (
        param_type.is_pointer() && param_type.subtype() == char_type() &&
        (arg.type().is_signedbv() || arg.type().is_unsignedbv()) &&
        arg.type() == char_type())
      {
        // Create a single-element array to hold the character
        typet char_array_type =
          array_typet(char_type(), from_integer(2, size_type()));

        // Create a temporary variable to hold the array
        symbolt &temp_symbol = converter_.create_tmp_symbol(
          call_,
          "$char_to_str$",
          char_array_type,
          exprt()); // No initial value here

        // Declare the temporary in the current block
        code_declt temp_decl(symbol_expr(temp_symbol));
        temp_decl.location() = location;
        converter_.current_block->copy_to_operands(temp_decl);

        // Assign the character to the first element
        exprt temp_array = symbol_expr(temp_symbol);
        exprt first_element =
          index_exprt(temp_array, from_integer(0, size_type()));
        code_assignt assign_char(first_element, arg);
        assign_char.location() = location;
        converter_.current_block->copy_to_operands(assign_char);

        // Assign null terminator to the second element
        exprt second_element =
          index_exprt(temp_array, from_integer(1, size_type()));
        code_assignt assign_null(second_element, from_integer(0, char_type()));
        assign_null.location() = location;
        converter_.current_block->copy_to_operands(assign_null);

        // Take address of the temporary variable
        arg = address_of_exprt(symbol_expr(temp_symbol));
      }

      // Check if parameter is an Optional type
      if (param_type.is_struct())
      {
        const struct_typet &struct_type = to_struct_type(param_type);
        std::string tag = struct_type.tag().as_string();

        if (tag.starts_with("tag-Optional_"))
        {
          // Wrap the argument in Optional struct
          arg = converter_.wrap_in_optional(arg, param_type);
        }
      }
    }

    // Handle string literal constants
    // Ensure they are proper null-terminated arrays
    if (arg_node["_type"] == "Constant" && arg_node["value"].is_string())
    {
      std::string str_value = arg_node["value"].get<std::string>();
      arg = converter_.get_string_builder().build_string_literal(str_value);
    }

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
        converter_.find_symbol("c:@F@__ESBMC_list_size");
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

      // Set the type to the return type of the function
      const exprt &func_expr = arg.op1();
      bool is_constructor = false;
      typet return_type;

      if (func_expr.is_symbol())
      {
        const symbolt *func_symbol =
          converter_.ns.lookup(to_symbol_expr(func_expr));
        if (func_symbol != nullptr)
        {
          const code_typet &func_type = to_code_type(func_symbol->type);
          return_type = func_type.return_type();

          // Special handling for constructors
          if (return_type.id() == "constructor")
          {
            is_constructor = true;
            // For constructors, use the class type instead of "constructor"
            return_type =
              type_handler_.get_typet(func_symbol->name.as_string());
          }

          func_call.type() = return_type;
        }
      }

      // Strip temporary $ctor_self$ parameters when constructors are used as
      // arguments (e.g., foo(Positive(2))). goto_sideeffects will add the
      // correct self parameter later.
      if (is_constructor)
      {
        exprt::operandst temp_args(
          args_expr.operands().begin(), args_expr.operands().end());
        func_call.arguments() = strip_ctor_self_parameters(temp_args);
      }
      else
      {
        // For non-constructors, just copy arguments
        for (const auto &operand : args_expr.operands())
          func_call.arguments().push_back(operand);
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

    arg_index++;
  }

  // Add default arguments for missing parameters
  size_t num_provided_args = call_["args"].size();
  size_t total_params = params.size();

  // Calculate how many arguments will actually be present after implicit additions
  size_t num_actual_args = num_provided_args;

  // For ClassMethod with no explicit args, check if object will be implicitly added
  if (function_type_ == FunctionType::ClassMethod && call_["args"].empty())
  {
    // Check the exact conditions where the object is added as an argument
    bool will_add_object = false;

    if (obj_symbol)
    {
      std::string var_type =
        type_handler_.get_var_type(obj_symbol->name.as_string());

      if (var_type == "int")
        will_add_object = true;
    }

    if (call_["func"]["value"]["_type"] == "BinOp")
      will_add_object = true;

    if (will_add_object)
      num_actual_args = 1;
  }

  // Check if we should skip validation for numpy functions
  // Numpy stub files often have incomplete/incorrect default parameter info
  // TODO: we have to revisit the function signature handling for numpy functions
  bool skip_validation = false;

  if (call_["func"]["_type"] == "Attribute")
  {
    auto current = call_["func"]["value"];

    // Walk up the attribute chain to get full module path
    while (current["_type"] == "Attribute")
      current = current["value"];

    if (current["_type"] == "Name")
    {
      std::string root = current["id"].get<std::string>();

      // Check if it starts with np or numpy
      skip_validation = (root == "np" || root == "numpy");
    }
  }

  // Check which parameters are already provided (by position or keyword)
  // and fill missing parameters with default values
  std::vector<bool> provided_params(total_params, false);

  // Mark positional arguments as provided
  for (size_t i = 0; i < num_actual_args && (i + param_offset) < total_params;
       ++i)
  {
    provided_params[i + param_offset] = true;
  }

  // For constructors, self is always implicitly provided
  if (function_type_ == FunctionType::Constructor && total_params > 0)
  {
    provided_params[0] = true;
  }

  // Mark keyword arguments as provided
  if (call_.contains("keywords") && call_["keywords"].is_array())
  {
    for (const auto &kw : call_["keywords"])
    {
      if (kw.contains("arg") && !kw["arg"].is_null())
      {
        std::string kw_name = kw["arg"].get<std::string>();
        for (size_t i = 0; i < params.size(); ++i)
        {
          if (params[i].get_base_name().as_string() == kw_name)
          {
            provided_params[i] = true;
            break;
          }
        }
      }
    }
  }

  // Validate required parameters and fill missing parameters with default values
  for (size_t param_idx = param_offset; param_idx < total_params; ++param_idx)
  {
    if (!provided_params[param_idx])
    {
      const auto &param_info = params[param_idx];

      // Check if parameter has a default value
      if (param_info.has_default_value())
      {
        exprt default_val = param_info.default_value();

        // Handle string default values: ensure they are properly initialized
        // Check if default value is a string array type
        if (
          default_val.type().is_array() &&
          default_val.type().subtype() == char_type())
        {
          // For constant string arrays, convert to string_constantt to ensure
          // proper initialization
          if (default_val.is_constant() || default_val.id() == "constant")
          {
            // Use existing string extraction utility
            std::string str_content =
              converter_.get_string_handler()
                .extract_string_from_array_operands(default_val);

            // Create string_constantt with proper type
            if (!str_content.empty() || default_val.operands().empty())
            {
              typet string_type = default_val.type();
              default_val = string_constantt(
                str_content, string_type, string_constantt::k_default);
            }
          }
        }

        // Handle Optional types: wrap default in Optional if needed
        if (param_info.type().is_struct())
        {
          const struct_typet &struct_type = to_struct_type(param_info.type());
          std::string tag = struct_type.tag().as_string();

          if (tag.starts_with("tag-Optional_"))
            default_val =
              converter_.wrap_in_optional(default_val, param_info.type());
        }

        // Convert array to pointer if parameter type is pointer
        // This matches the behavior for positional arguments (line 2470-2480)
        const typet &param_type = param_info.type();
        if (default_val.type().is_array() && param_type.is_pointer())
        {
          // For string constants, use address_of
          if (default_val.id() == "string-constant")
          {
            default_val = address_of_exprt(default_val);
          }
          else
          {
            // For regular arrays, get base address
            default_val =
              converter_.get_string_handler().get_array_base_address(
                default_val);
          }
        }

        // Ensure arguments vector is large enough
        if (call.arguments().size() <= param_idx)
          call.arguments().resize(param_idx + 1);

        call.arguments()[param_idx] = default_val;
      }
      else if (!skip_validation)
      {
        // Parameter is missing and has no default value - this is an error
        std::string param_name = param_info.get_base_name().as_string();
        std::string func_name = function_id_.get_function();

        std::ostringstream msg;
        msg << func_name << "() missing required positional argument: '"
            << param_name << "'";

        exprt exception =
          converter_.get_exception_handler().gen_exception_raise(
            "TypeError", msg.str());
        locationt loc = converter_.get_location_from_decl(call_);
        exception.location() = loc;
        exception.location().user_provided(true);

        return exception;
      }
    }
  }

  // For constructors without current_lhs, create temp var and add self if needed
  // Note: get_return_statements() will handle return statements separately
  if (function_type_ == FunctionType::Constructor && !converter_.current_lhs)
  {
    size_t num_provided_args = call_["args"].size();

    // Only add self if arguments size matches user args (no self added yet)
    if (call.arguments().size() == num_provided_args)
    {
      // Self parameter not added yet - this is a standalone call (e.g., Positive(2))
      // Create temporary object as self parameter
      typet class_type = type_handler_.get_typet(func_symbol->name.as_string());
      symbolt &temp_self =
        converter_.create_tmp_symbol(call_, "$ctor_self$", class_type, exprt());
      converter_.symbol_table().add(temp_self);

      // Add declaration for temporary object
      code_declt temp_decl(symbol_expr(temp_self));
      temp_decl.location() = location;
      converter_.current_block->copy_to_operands(temp_decl);

      // Insert self as first argument
      call.arguments().insert(
        call.arguments().begin(), gen_address_of(symbol_expr(temp_self)));
    }
  }

  return call;
}

exprt::operandst
function_call_expr::strip_ctor_self_parameters(const exprt::operandst &args)
{
  exprt::operandst new_args;
  for (const auto &arg : args)
  {
    bool is_ctor_self = false;
    if (arg.is_address_of() && !arg.operands().empty() && arg.op0().is_symbol())
    {
      const std::string &arg_id = arg.op0().identifier().as_string();
      if (arg_id.find("$ctor_self$") != std::string::npos)
      {
        is_ctor_self = true;
      }
    }
    if (!is_ctor_self)
    {
      new_args.push_back(arg);
    }
  }
  return new_args;
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
  const nlohmann::json &args,
  const nlohmann::json &keywords) const
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

  auto types_match = [&](const typet &expected, const typet &actual) {
    return base_type_eq(expected, actual, converter_.ns) ||
           (type_utils::is_string_type(expected) &&
            type_utils::is_string_type(actual));
  };

  for (size_t i = 0; i < args.size(); ++i)
  {
    size_t param_idx = i + param_offset;
    if (param_idx >= params.size())
      break;

    exprt arg = converter_.get_expr(args[i]);
    const typet &expected_type = params[param_idx].type();
    const typet &actual_type = arg.type();

    // Check for type mismatch
    if (!types_match(expected_type, actual_type))
    {
      std::string expected_str = type_handler_.type_to_string(expected_type);
      std::string actual_str = type_handler_.type_to_string(actual_type);

      std::ostringstream msg;
      msg << "TypeError: Argument " << (i + 1) << " has incompatible type '"
          << actual_str << "'; expected '" << expected_str << "'";

      exprt exception = converter_.get_exception_handler().gen_exception_raise(
        "TypeError", msg.str());

      // Add location information from the call
      locationt loc = converter_.get_location_from_decl(call_);
      exception.location() = loc;
      exception.location().user_provided(true);

      return exception;
    }
  }

  if (keywords.is_array())
  {
    for (const auto &keyword : keywords)
    {
      if (!keyword.contains("arg") || keyword["arg"].is_null())
        continue;

      const std::string &param_name = keyword["arg"].get<std::string>();

      size_t param_idx = params.size();
      for (size_t i = param_offset; i < params.size(); ++i)
      {
        if (params[i].get_base_name().as_string() == param_name)
        {
          param_idx = i;
          break;
        }
      }

      if (param_idx >= params.size() || !keyword.contains("value"))
        continue;

      exprt arg = converter_.get_expr(keyword["value"]);
      const typet &expected_type = params[param_idx].type();
      const typet &actual_type = arg.type();

      if (!types_match(expected_type, actual_type))
      {
        std::string expected_str = type_handler_.type_to_string(expected_type);
        std::string actual_str = type_handler_.type_to_string(actual_type);

        std::ostringstream msg;
        msg << "TypeError: Argument '" << param_name
            << "' has incompatible type '" << actual_str << "'; expected '"
            << expected_str << "'";

        exprt exception =
          converter_.get_exception_handler().gen_exception_raise(
            "TypeError", msg.str());

        locationt loc = converter_.get_location_from_decl(call_);
        exception.location() = loc;
        exception.location().user_provided(true);

        return exception;
      }
    }
  }

  return nil_exprt();
}
