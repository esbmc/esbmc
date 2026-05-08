#include <python-frontend/function_call_expr.h>
#include <python-frontend/cmath_lowering_policy.h>
#include <python-frontend/complex_handler.h>
#include <python-frontend/complex_handler_utils.h>
#include <python-frontend/exception_utils.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/math_guard_utils.h>
#include <python-frontend/string_handler_utils.h>
#include <python-frontend/python_exception_handler.h>
#include <python-frontend/python_list.h>
#include <python-frontend/string_builder.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/tuple_handler.h>
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
#include <algorithm>
#include <cmath>
#include <cctype>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unordered_set>

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

/// Extract the mandatory "_type" field from a Python AST JSON node.
/// Throws if the node is not an object or lacks a valid "_type" string.
std::string node_type_of(const nlohmann::json &node)
{
  if (
    !node.is_object() || !node.contains("_type") || !node["_type"].is_string())
  {
    throw std::runtime_error("Missing or invalid _type field in AST node");
  }
  return node["_type"].get<std::string>();
}

bool is_cpp_throw_expr(const exprt &e)
{
  return e.statement() == "cpp-throw";
}

double round_ties_to_even(const double value)
{
  const double lower = std::floor(value);
  const double diff = value - lower;
  constexpr double tie_eps = 1e-12;

  if (diff < 0.5 - tie_eps)
    return lower;
  if (diff > 0.5 + tie_eps)
    return lower + 1.0;

  const double parity = std::fmod(std::fabs(lower), 2.0);
  const bool lower_is_even =
    parity < tie_eps || std::fabs(parity - 2.0) < tie_eps;
  return lower_is_even ? lower : lower + 1.0;
}

double round_to_ndigits_ties_even(const double value, const int ndigits)
{
  auto round_ld_ties_even = [](const long double v) -> long double {
    const long double lower = std::floor(v);
    const long double diff = v - lower;
    constexpr long double tie_eps = 1e-15L;

    if (diff < 0.5L - tie_eps)
      return lower;
    if (diff > 0.5L + tie_eps)
      return lower + 1.0L;

    const long double parity = std::fmod(std::fabs(lower), 2.0L);
    const bool lower_is_even =
      parity < tie_eps || std::fabs(parity - 2.0L) < tie_eps;
    return lower_is_even ? lower : lower + 1.0L;
  };

  // Keep scaling deterministic across libm implementations.
  long double scale = 1.0L;
  if (ndigits >= 0)
  {
    for (int i = 0; i < ndigits; ++i)
      scale *= 10.0L;
    return static_cast<double>(
      round_ld_ties_even(static_cast<long double>(value) * scale) / scale);
  }

  for (int i = 0; i < -ndigits; ++i)
    scale *= 10.0L;
  return static_cast<double>(
    round_ld_ties_even(static_cast<long double>(value) / scale) * scale);
}
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

const symbolt *
function_call_expr::cached_find_symbol(const std::string &id) const
{
  auto it = sym_cache_.find(id);
  if (it != sym_cache_.end())
    return it->second;
  const symbolt *sym = converter_.find_symbol(id);
  if (sym)
    sym_cache_.emplace(id, sym);
  return sym;
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

  const auto &func_node = call_["func"];
  if (
    !func_node.contains("_type") || !func_node["_type"].is_string() ||
    func_node["_type"] != "Attribute")
  {
    return;
  }

  if (!func_node.contains("value") || !func_node["value"].is_object())
    return;

  std::string caller = get_object_name();
  const auto &func_value = func_node["value"];

  // Check for nested instance attribute (e.g., self.b.a.method())
  // Exclude module.Class.method() pattern
  // Walk the full attribute chain to find the root Name node, regardless of depth.
  bool is_nested_instance_attr = false;
  if (node_type_of(func_value) == "Attribute")
  {
    const nlohmann::json *cur = &func_value;
    while (node_type_of(*cur) == "Attribute")
    {
      if (!cur->contains("value") || !(*cur)["value"].is_object())
        break;
      cur = &(*cur)["value"];
    }
    if (
      node_type_of(*cur) == "Name" && cur->contains("id") &&
      (*cur)["id"].is_string())
    {
      std::string root_name = (*cur)["id"].get<std::string>();
      if (!converter_.is_imported_module(root_name))
        is_nested_instance_attr = true;
    }
  }

  // Detect A().f(...): method call on a temporary instance.
  // This covers both direct construction (A().f()) and chained method calls
  // (B().g().f()) — any Call node in the value position means the receiver
  // is a temporary, so we must treat it as an InstanceMethod regardless of
  // whether the inferred receiver class name matches a class in the AST.
  bool obj_is_temp_instance = false;
  if (
    node_type_of(func_value) == "Call" && func_value.contains("func") &&
    func_value["func"].is_object())
  {
    const std::string callee_type = node_type_of(func_value["func"]);
    obj_is_temp_instance = callee_type == "Name" || callee_type == "Attribute";
  }

  // Handling a function call as a class method call when:
  // (1) The caller corresponds to a class name, for example: MyClass.foo().
  // (2) Calling methods of built-in types, such as int.from_bytes()
  //     All the calls to built-in methods are handled by class methods in operational models.
  // (3) Calling a instance method from a built-in type object, for example: x.bit_length() when x is an int
  // If the caller is a class or a built-in type, the following condition detects a class method call.
  if (
    !is_nested_instance_attr && !obj_is_temp_instance &&
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

bool function_call_expr::is_nondet_call() const
{
  static std::regex pattern(
    R"(nondet_(int|char|bool|float|str|complex)|__VERIFIER_nondet_(int|char|bool|float|str|complex))");

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

  // Record the companion $input_len$ symbol so len() on this string (or any
  // variable aliasing it) can return the symbolic length directly instead of
  // falling back to strlen() loop-unrolling.
  converter_.input_str_to_len_sym_[input_sym.id.as_string()] =
    len_sym.id.as_string();

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

  if (type == "complex")
  {
    // nondet_complex() → make_complex(nondet_real, nondet_imag)
    const typet &dt = cached_double_type();
    exprt nondet_real("sideeffect", dt);
    nondet_real.statement("nondet");
    exprt nondet_imag("sideeffect", dt);
    nondet_imag.statement("nondet");
    return make_complex(nondet_real, nondet_imag);
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

    // String special case: get_typet("str") returns char[0], which the
    // generic encoding lowers to gen_zero(char[0]) — an empty array
    // constant. simplify_python_builtins then has to recover the answer
    // from value-set state, which is fragile (e.g. depends on operational
    // model layout). When obj's static type is already a char-array or
    // char-pointer (a Python str), short-circuit to true here.
    if (
      expected_type.is_array() &&
      to_array_type(expected_type).subtype() == char_type())
    {
      const typet &obj_type = obj_expr.type();
      if (
        (obj_type.is_array() && obj_type.subtype() == char_type()) ||
        (obj_type.is_pointer() && obj_type.subtype() == char_type()))
        return true_exprt();
    }

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

exprt function_call_expr::handle_type_call() const
{
  const auto &args = call_["args"];
  if (args.size() != 1)
    throw std::runtime_error("type() requires exactly 1 argument");

  exprt arg_expr = converter_.get_expr(args[0]);
  std::string type_name = type_handler_.get_python_type_name(arg_expr.type());
  if (type_name.empty())
    type_name = arg_expr.type().id_string();

  typet str_type = type_handler_.build_array(char_type(), type_name.size() + 1);
  return constant_exprt(type_name, type_name, str_type);
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

exprt function_call_expr::handle_int_to_bytes() const
{
  const auto &args = call_["args"];
  // Python accepts both int.to_bytes(x, ...) and x.to_bytes(...).
  const bool is_type_method_call = call_["func"]["_type"] == "Attribute" &&
                                   call_["func"]["value"]["_type"] == "Name" &&
                                   call_["func"]["value"]["id"] == "int";

  // In the type-method form, the integer value is passed explicitly as the
  // first argument. In the instance-method form, it comes from the receiver.
  if (
    args.size() < (is_type_method_call ? 3 : 2) ||
    args.size() > (is_type_method_call ? 4 : 3))
  {
    throw std::runtime_error(
      is_type_method_call ? "int.to_bytes() expects 3 or 4 positional "
                            "arguments"
                          : "int.to_bytes() expects 2 or 3 positional "
                            "arguments");
  }

  exprt value = is_type_method_call
                  ? converter_.get_expr(args[0])
                  : converter_.get_expr(call_["func"]["value"]);
  const nlohmann::json &length_arg = args[is_type_method_call ? 1 : 0];
  const nlohmann::json &byteorder_arg = args[is_type_method_call ? 2 : 1];

  if (
    !length_arg.contains("value") || !length_arg["value"].is_number_unsigned())
    throw std::runtime_error(
      "int.to_bytes() currently expects a constant unsigned length");

  const std::size_t length = length_arg["value"].get<std::size_t>();

  bool big_endian = true;
  if (byteorder_arg.contains("value"))
  {
    if (byteorder_arg["value"].is_boolean())
      big_endian = byteorder_arg["value"].get<bool>();
    else if (byteorder_arg["value"].is_string())
      big_endian = byteorder_arg["value"].get<std::string>() == "big";
  }

  const typet bytes_type = type_handler_.get_typet("bytes", length);
  exprt result = gen_zero(bytes_type);
  const typet &elem_type = bytes_type.subtype();

  if (!value.type().is_unsignedbv())
  {
    // Convert the source value to an unsigned integer type before extracting
    // individual bytes with shifts and masks.
    const unsigned width =
      (value.type().is_signedbv() || value.type().is_unsignedbv())
        ? std::max(1u, bv_width(value.type()))
        : 64;
    value = typecast_exprt(value, unsignedbv_typet(width));
  }

  // Fill the output array one byte at a time. For big-endian we start from the
  // most significant byte; for little-endian we start from the least significant one.
  for (std::size_t i = 0; i < length; ++i)
  {
    const std::size_t byte_index = big_endian ? (length - 1 - i) : i;

    // Shift the selected byte down to the low 8 bits and mask everything else out.
    const exprt shift_amount = from_integer(byte_index * 8, value.type());
    exprt shifted("shr", value.type());
    shifted.copy_to_operands(value, shift_amount);

    exprt masked("bitand", value.type());
    masked.copy_to_operands(shifted, from_integer(0xff, value.type()));

    result.operands().at(i) = typecast_exprt(masked, elem_type);
  }

  return result;
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

static std::string format_double_for_complex(double d)
{
  // Use snprintf instead of std::to_string to avoid locale dependence
  // (std::to_string may emit ',' instead of '.' in some locales).
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%f", d);
  std::string s(buf);
  // Remove trailing zeros: "5.500000" → "5.5"
  s.erase(s.find_last_not_of('0') + 1, std::string::npos);
  if (s.back() == '.')
    s.pop_back();
  return s;
}

exprt function_call_expr::handle_complex_to_str() const
{
  // Non-constant complex: fall back to a generic placeholder.
  return converter_.get_string_builder().build_string_literal("(complex)");
}

// Try to extract constant real and imaginary parts from a JSON node
// that represents a complex value. Returns true and sets real_val/imag_val
// if the values can be extracted statically.
static bool try_extract_complex_parts_from_json(
  const nlohmann::json &arg,
  const nlohmann::json &ast,
  const std::string &current_func,
  double &real_val,
  double &imag_val)
{
  // Helper: extract a numeric literal from a JSON node, supporting
  // Constant nodes, UnaryOp(USub/UAdd, Constant), etc.
  std::function<std::optional<double>(const nlohmann::json &)> get_numeric;
  get_numeric = [&](const nlohmann::json &node) -> std::optional<double> {
    if (!node.contains("_type"))
      return std::nullopt;

    const auto &type = node["_type"];

    // Direct Constant node.
    if (type == "Constant" && node.contains("value"))
    {
      const auto &val = node["value"];
      if (val.is_number_integer())
        return static_cast<double>(val.get<int64_t>());
      if (val.is_number_float())
        return val.get<double>();
      return std::nullopt;
    }

    // UnaryOp: -x or +x
    if (type == "UnaryOp" && node.contains("op") && node.contains("operand"))
    {
      const auto &op = node["op"]["_type"];
      auto operand = get_numeric(node["operand"]);
      if (!operand)
        return std::nullopt;
      if (op == "USub")
        return -(*operand);
      if (op == "UAdd")
        return *operand;
    }

    return std::nullopt;
  };

  // Case 1: arg is a Call to complex() with literal numeric arguments.
  if (
    arg.contains("_type") && arg["_type"] == "Call" && arg.contains("func") &&
    arg["func"].contains("id") && arg["func"]["id"] == "complex")
  {
    const auto &args =
      arg.contains("args") ? arg["args"] : nlohmann::json::array();
    real_val = 0.0;
    imag_val = 0.0;
    if (args.size() >= 1)
    {
      auto r = get_numeric(args[0]);
      if (!r)
        return false;
      real_val = *r;
    }
    if (args.size() >= 2)
    {
      auto i = get_numeric(args[1]);
      if (!i)
        return false;
      imag_val = *i;
    }
    return true;
  }

  // Case 2: arg is a Name referencing a variable assigned to complex().
  if (arg.contains("_type") && arg["_type"] == "Name" && arg.contains("id"))
  {
    const std::string &var_name = arg["id"].get_ref<const std::string &>();
    nlohmann::json var_decl =
      json_utils::find_var_decl(var_name, current_func, ast);
    if (!var_decl.empty() && var_decl.contains("value"))
    {
      return try_extract_complex_parts_from_json(
        var_decl["value"], ast, current_func, real_val, imag_val);
    }
  }

  return false;
}

static std::string format_complex_string(double real_val, double imag_val)
{
  // Format according to Python's complex repr rules:
  //   real == 0 → "{imag}j"     (e.g., "2j", "-1j", "0j")
  //   real != 0 → "({real}{sign}{imag}j)"  (e.g., "(1+2j)", "(1-2j)")
  std::string result;
  std::string imag_str = format_double_for_complex(std::abs(imag_val));

  // Python distinguishes -0.0 from 0.0: complex(-0.0, 1) → "(-0+1j)".
  // IEEE 754: -0.0 == 0.0, so we must check the sign bit explicitly.
  const bool real_is_zero = real_val == 0.0 && !std::signbit(real_val);

  if (real_is_zero)
  {
    if (imag_val < 0.0)
      result = "-" + imag_str + "j";
    else
      result = imag_str + "j";
  }
  else
  {
    std::string real_str = format_double_for_complex(real_val);
    std::string sign = (imag_val >= 0.0) ? "+" : "-";
    result = "(" + real_str + sign + imag_str + "j)";
  }

  return result;
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

  {
    char *end = nullptr;
    double dval = std::strtod(value_opt->c_str(), &end);
    if (!end || end != value_opt->c_str() + value_opt->size())
    {
      log_error(
        "Failed float conversion from string \"{}\": invalid argument",
        *value_opt);
      return from_double(0.0, type_handler_.get_typet("float", 0));
    }
    return from_double(dval, type_handler_.get_typet("float", 0));
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
      func_name != "bool" && func_name != "bytes" && func_name != "complex")
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

      if (is_complex_type(inferred_type))
        return converter_.get_complex_handler().handle_abs(inferred_expr);

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

      if (is_complex_type(operand_type))
        return converter_.get_complex_handler().handle_abs(operand_expr);

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

  exprt fallback_expr = converter_.get_expr(arg);
  if (fallback_expr.is_nil() || fallback_expr.statement() == "cpp-throw")
    return fallback_expr;

  if (is_complex_type(fallback_expr.type()))
    return converter_.get_complex_handler().handle_abs(fallback_expr);

  exprt abs_expr("abs", fallback_expr.type());
  abs_expr.copy_to_operands(fallback_expr);
  return abs_expr;
}

exprt function_call_expr::handle_round(nlohmann::json &arg) const
{
  const auto &args = call_["args"];
  bool has_ndigits = args.size() >= 2;

  // Reject strings early
  if (is_string_arg(arg))
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError", "type str doesn't define __round__ method");

  // Handle unary minus (e.g., round(-3.6))
  // Unlike abs(), round() must preserve the sign.
  bool is_negated = false;
  if (arg.contains("_type") && arg["_type"] == "UnaryOp")
  {
    const auto &op = arg["op"];
    const auto &operand = arg["operand"];
    if (op["_type"] == "USub" && operand.contains("value"))
    {
      arg = operand;
      is_negated = true;
    }
  }

  // Compile-time evaluation for numeric literals
  if (arg.contains("value") && arg["value"].is_number())
  {
    if (!has_ndigits)
    {
      // round(x) -> nearest integer (returns int)
      if (arg["value"].is_number_integer())
      {
        int val = arg["value"].get<int>();
        if (is_negated)
          val = -val;
        arg["value"] = val;
        arg["type"] = "int";
      }
      else
      {
        double val = arg["value"].get<double>();
        if (is_negated)
          val = -val;
        arg["value"] = static_cast<int>(round_ties_to_even(val));
        arg["type"] = "int";
      }
      typet t = type_handler_.get_typet("int", 0);
      exprt expr = converter_.get_expr(arg);
      expr.type() = t;
      return expr;
    }
    else
    {
      // round(x, n) -> float rounded to n decimals
      auto ndigits_arg = args[1];
      if (
        ndigits_arg.contains("value") &&
        ndigits_arg["value"].is_number_integer())
      {
        int n = ndigits_arg["value"].get<int>();
        double val = arg["value"].is_number_integer()
                       ? static_cast<double>(arg["value"].get<int>())
                       : arg["value"].get<double>();
        if (is_negated)
          val = -val;
        double rounded = round_to_ndigits_ties_even(val, n);
        arg["value"] = rounded;
        arg["type"] = "float";
        typet t = type_handler_.get_typet("float", 0);
        exprt expr = converter_.get_expr(arg);
        expr.type() = t;
        return expr;
      }
    }
  }

  // Symbolic: try to build an expression for round(x)
  if (!has_ndigits)
  {
    try
    {
      exprt operand_expr = converter_.get_expr(arg);
      typet float_type = type_handler_.get_typet("float", 0);
      typet int_type = type_handler_.get_typet("int", 0);

      // Use nearbyint (round-to-nearest-even) on the float operand,
      // then typecast to int — matching Python's round() semantics.
      exprt nearbyint_expr("nearbyint", float_type);
      nearbyint_expr.copy_to_operands(operand_expr);
      return typecast_exprt(nearbyint_expr, int_type);
    }
    catch (const std::exception &)
    {
      return converter_.get_exception_handler().gen_exception_raise(
        "TypeError", "failed to infer operand type for round()");
    }
  }

  log_warning("round() with symbolic arguments not fully supported");
  return nil_exprt();
}

exprt function_call_expr::handle_complex() const
{
  // Ensure complex type symbol exists for downstream attribute resolution.
  (void)type_handler_.get_typet(std::string("complex"));

  const nlohmann::json &arguments =
    call_.contains("args") ? call_["args"] : nlohmann::json::array();
  const nlohmann::json &keywords =
    call_.contains("keywords") ? call_["keywords"] : nlohmann::json::array();

  auto raise_type_error = [this](const std::string &msg) -> exprt {
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError", msg);
  };
  auto raise_value_error = [this](const std::string &msg) -> exprt {
    return converter_.get_exception_handler().gen_exception_raise(
      "ValueError", msg);
  };
  auto zero = []() -> exprt { return from_double(0.0, double_type()); };
  auto normalize_numeric_expr_for_complex = [this](exprt value) -> exprt {
    return converter_.get_complex_handler().normalize_numeric_expr(value);
  };
  auto is_cpp_throw = [](const exprt &e) -> bool {
    return e.statement() == "cpp-throw";
  };
  auto extract_constant_string =
    [&](const nlohmann::json &arg, std::string &out) -> bool {
    if (!arg.contains("value"))
      return false;
    if (!arg["value"].is_string())
      return false;
    out = arg["value"].get<std::string>();
    return true;
  };
  auto is_bytes_literal = [](const nlohmann::json &arg) -> bool {
    if (arg.contains("encoded_bytes"))
      return true;
    if (
      arg.contains("annotation") && arg["annotation"].contains("id") &&
      arg["annotation"]["id"] == "bytes")
      return true;
    if (
      arg.contains("esbmc_type_annotation") &&
      arg["esbmc_type_annotation"] == "bytes")
      return true;
    if (arg.contains("kind") && arg["kind"] == "bytes")
      return true;
    return false;
  };
  auto is_bytearray_call = [](const nlohmann::json &arg) -> bool {
    return (
      arg.contains("_type") && arg["_type"] == "Call" && arg.contains("func") &&
      arg["func"].contains("_type") && arg["func"]["_type"] == "Name" &&
      arg["func"].contains("id") && arg["func"]["id"] == "bytearray");
  };
  auto byteslike_name = [&](const nlohmann::json &arg) -> std::string {
    if (is_bytes_literal(arg))
      return "bytes";
    if (is_bytearray_call(arg))
      return "bytearray";
    return "";
  };
  auto is_bytes_annotated_name = [&](const nlohmann::json &arg) -> bool {
    if (!(arg.contains("_type") && arg["_type"] == "Name" &&
          arg.contains("id")))
      return false;

    const std::string var_name = arg["id"].get<std::string>();
    nlohmann::json var_decl = json_utils::find_var_decl(
      var_name, converter_.current_function_name(), converter_.ast());
    if (
      var_decl.empty() || !var_decl.contains("annotation") ||
      var_decl["annotation"].is_null())
      return false;

    const auto &annotation = var_decl["annotation"];
    return annotation.contains("id") && annotation["id"] == "bytes";
  };
  auto try_dispatch_numeric_dunder =
    [&](const std::string &op, exprt &operand) -> exprt {
    return converter_.dispatch_unary_dunder_operator(
      op, operand, converter_.get_location_from_decl(call_));
  };
  auto try_convert_via_numeric_dunders =
    [&](exprt &value, bool require_complex_result) -> std::optional<exprt> {
    exprt complex_result = try_dispatch_numeric_dunder("complex", value);
    if (!complex_result.is_nil())
    {
      if (is_cpp_throw(complex_result))
        return complex_result;
      if (!is_complex_type(complex_result.type()))
      {
        if (require_complex_result)
          return raise_type_error("__complex__ returned non-complex");
        complex_result = promote_to_complex(complex_result);
      }
      return complex_result;
    }

    exprt float_result = try_dispatch_numeric_dunder("float", value);
    if (!float_result.is_nil())
    {
      if (is_cpp_throw(float_result))
        return float_result;
      const typet &float_t = float_result.type();
      const bool is_python_float =
        float_t == double_type() ||
        (float_t.is_floatbv() && to_floatbv_type(float_t).get_width() == 64);
      if (!is_python_float)
        return raise_type_error("__float__ returned non-float");
      return make_complex(float_result, zero());
    }

    exprt index_result = try_dispatch_numeric_dunder("index", value);
    if (!index_result.is_nil())
    {
      if (is_cpp_throw(index_result))
        return index_result;
      const typet &index_t = index_result.type();
      const bool is_python_int =
        index_t.is_signedbv() || index_t.is_unsignedbv() || index_t.is_bool();
      if (!is_python_int)
        return raise_type_error("__index__ returned non-int");
      if (index_result.type() != double_type())
        index_result = typecast_exprt(index_result, double_type());
      return make_complex(index_result, zero());
    }

    return std::nullopt;
  };
  auto is_unsigned_byte_array = [](const typet &type) -> bool {
    if (!type.is_array())
      return false;
    const typet &subtype = type.subtype();
    return subtype.is_unsignedbv() &&
           to_unsignedbv_type(subtype).get_width() == 8;
  };

  if (arguments.size() > 2)
    return raise_type_error("complex() takes at most 2 arguments");

  const nlohmann::json *real_json = nullptr;
  const nlohmann::json *imag_json = nullptr;

  if (!arguments.empty())
    real_json = &arguments[0];
  if (arguments.size() >= 2)
    imag_json = &arguments[1];

  if (keywords.is_array() && !keywords.empty())
  {
    try
    {
      auto kw_vals =
        string_call_utils::collect_keyword_values("complex", keywords);
      string_call_utils::ensure_allowed_keywords(
        "complex", kw_vals, {"real", "imag"});

      if (
        auto *real_kw = string_call_utils::find_keyword_value(kw_vals, "real"))
      {
        if (real_json != nullptr)
          return raise_type_error(
            "complex() got multiple values for argument 'real'");
        real_json = real_kw;
      }
      if (
        auto *imag_kw = string_call_utils::find_keyword_value(kw_vals, "imag"))
      {
        if (imag_json != nullptr)
          return raise_type_error(
            "complex() got multiple values for argument 'imag'");
        imag_json = imag_kw;
      }
    }
    catch (const std::runtime_error &e)
    {
      return raise_type_error(e.what());
    }
  }

  nlohmann::json default_real_json;
  if (real_json == nullptr)
  {
    default_real_json = {{"_type", "Constant"}, {"value", 0}};
    real_json = &default_real_json;
  }

  if (imag_json == nullptr)
  {
    // bytes/bytearray are rejected by CPython in complex(x) one-arg form.
    const std::string real_byteslike = byteslike_name(*real_json);
    if (!real_byteslike.empty())
      return raise_type_error(
        "complex() first argument must be a string or a number, not '" +
        real_byteslike + "'");
    if (is_bytes_annotated_name(*real_json))
      return raise_type_error(
        "complex() first argument must be a string or a number, not 'bytes'");

    // One-argument form accepts string literals.
    std::string text;
    if (extract_constant_string(*real_json, text))
    {
      double real = 0.0, imag = 0.0;
      if (!complex_utils::parse_complex_string(text, real, imag))
        return raise_value_error("complex() arg is a malformed string");
      return make_complex(
        from_double(real, double_type()), from_double(imag, double_type()));
    }

    // Best-effort support for non-literal string symbols.
    if ((*real_json).contains("_type") && (*real_json)["_type"] == "Name")
    {
      const symbolt *sym = lookup_python_symbol((*real_json)["id"]);
      if (sym)
      {
        const typet &symbol_type =
          sym->value.type().is_not_nil() ? sym->value.type() : sym->type;
        if (
          symbol_type.is_array() && symbol_type.subtype().is_unsignedbv() &&
          to_unsignedbv_type(symbol_type.subtype()).get_width() == 8)
          return raise_type_error(
            "complex() first argument must be a string or a number, not "
            "'bytes'");

        // Non-text arrays (e.g., bytes variables represented as integer arrays)
        // are not valid real arguments for complex(x).
        if (symbol_type.is_array())
        {
          const typet &elem_type = symbol_type.subtype();
          const bool is_textual_char_array =
            elem_type == char_type() ||
            (elem_type.is_signedbv() &&
             to_signedbv_type(elem_type).get_width() == 8) ||
            (elem_type.is_unsignedbv() &&
             to_unsignedbv_type(elem_type).get_width() == 8);
          if (!is_textual_char_array)
            return raise_type_error(
              "complex() first argument must be a string or a number, not "
              "'bytes'");
        }

        const bool maybe_text_symbol =
          symbol_type.is_array() ||
          (symbol_type.is_signedbv() &&
           to_signedbv_type(symbol_type).get_width() == 8);
        if (maybe_text_symbol)
        {
          auto value_opt = extract_string_from_symbol(sym);
          if (value_opt)
          {
            double real = 0.0, imag = 0.0;
            if (!complex_utils::parse_complex_string(*value_opt, real, imag))
              return raise_value_error("complex() arg is a malformed string");
            return make_complex(
              from_double(real, double_type()),
              from_double(imag, double_type()));
          }
        }
      }
    }

    exprt value = converter_.get_expr(*real_json);
    if (value.is_nil() || is_cpp_throw(value))
      return value;

    if (is_unsigned_byte_array(value.type()))
      return raise_type_error(
        "complex() first argument must be a string or a number, not 'bytes'");
    if (value.type().is_array())
    {
      const typet &elem_type = value.type().subtype();
      const bool is_textual_char_array =
        elem_type == char_type() ||
        (elem_type.is_signedbv() &&
         to_signedbv_type(elem_type).get_width() == 8) ||
        (elem_type.is_unsignedbv() &&
         to_unsignedbv_type(elem_type).get_width() == 8);
      if (!is_textual_char_array)
        return raise_type_error(
          "complex() first argument must be a string or a number, not 'bytes'");
    }

    if (is_complex_type(value.type()))
      return value;

    if (std::optional<exprt> dunder_value =
          try_convert_via_numeric_dunders(value, true);
        dunder_value.has_value())
      return *dunder_value;

    value = normalize_numeric_expr_for_complex(value);
    if (is_cpp_throw_expr(value))
      return value;

    if (value.type() != double_type())
    {
      value = typecast_exprt(value, double_type());
    }

    return make_complex(value, zero());
  }

  // Two-argument form does not accept string / bytes / bytearray values.
  if (is_string_arg(*real_json))
    return raise_type_error(
      "complex() can't take second arg if first is a string");
  const std::string real_byteslike = byteslike_name(*real_json);
  if (!real_byteslike.empty() || is_bytes_annotated_name(*real_json))
    return raise_type_error(
      "complex() first argument must be a string or a number, not '" +
      (real_byteslike.empty() ? "bytes" : real_byteslike) + "'");
  if (
    is_string_arg(*imag_json) || !byteslike_name(*imag_json).empty() ||
    is_bytes_annotated_name(*imag_json))
    return raise_type_error("complex() second arg can't be a string");

  exprt real_arg = converter_.get_expr(*real_json);
  if (real_arg.is_nil() || is_cpp_throw(real_arg))
    return real_arg;

  exprt imag_arg = converter_.get_expr(*imag_json);
  if (imag_arg.is_nil() || is_cpp_throw(imag_arg))
    return imag_arg;

  if (
    is_unsigned_byte_array(real_arg.type()) ||
    is_unsigned_byte_array(imag_arg.type()))
    return raise_type_error("complex() second arg can't be a string");

  if (!is_complex_type(real_arg.type()))
  {
    if (std::optional<exprt> dunder_real =
          try_convert_via_numeric_dunders(real_arg, true);
        dunder_real.has_value())
    {
      if (is_cpp_throw(*dunder_real))
        return *dunder_real;
      real_arg = *dunder_real;
    }
  }
  real_arg = normalize_numeric_expr_for_complex(real_arg);
  if (is_cpp_throw_expr(real_arg))
    return real_arg;

  if (!is_complex_type(imag_arg.type()))
  {
    if (std::optional<exprt> dunder_imag =
          try_convert_via_numeric_dunders(imag_arg, true);
        dunder_imag.has_value())
    {
      if (is_cpp_throw(*dunder_imag))
        return *dunder_imag;
      imag_arg = *dunder_imag;
    }
  }
  imag_arg = normalize_numeric_expr_for_complex(imag_arg);
  if (is_cpp_throw_expr(imag_arg))
    return imag_arg;

  // Python semantics: complex(x, y) == x + y * 1j, including complex args.
  real_arg = promote_to_complex(real_arg);
  imag_arg = promote_to_complex(imag_arg);

  exprt a = member_exprt(real_arg, "real", double_type());
  exprt b = member_exprt(real_arg, "imag", double_type());
  exprt c = member_exprt(imag_arg, "real", double_type());
  exprt d = member_exprt(imag_arg, "imag", double_type());

  exprt real_part("ieee_sub", double_type());
  real_part.copy_to_operands(a, d);

  exprt imag_part("ieee_add", double_type());
  imag_part.copy_to_operands(b, c);

  return make_complex(real_part, imag_part);
}

exprt function_call_expr::build_constant_from_arg() const
{
  const std::string &func_name = function_id_.get_function();

  if (func_name == "complex")
    return handle_complex();

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

  // Handle str(z) / repr(z) where z is a complex expression.
  // Must check before the arg["value"]-based dispatch below, since
  // complex args come from Name/Call nodes without a "value" field.
  if (func_name == "str" || func_name == "repr")
  {
    // First try to extract complex parts directly from the JSON AST.
    double real_val = 0.0, imag_val = 0.0;
    if (try_extract_complex_parts_from_json(
          arg,
          converter_.ast(),
          converter_.current_function_name(),
          real_val,
          imag_val))
    {
      return converter_.get_string_builder().build_string_literal(
        format_complex_string(real_val, imag_val));
    }

    // Fall back: check if the expression has complex type.
    bool is_simple_literal =
      arg.contains("value") &&
      (arg["value"].is_number_integer() || arg["value"].is_number_float() ||
       arg["value"].is_string());
    if (!is_simple_literal)
    {
      exprt value_expr = converter_.get_expr(arg);
      if (
        !value_expr.is_nil() && value_expr.statement() != "cpp-throw" &&
        is_complex_type(value_expr.type()))
        return handle_complex_to_str();
    }
  }

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
      {
        const std::string raw_val = arg["value"].get<std::string>();
        char *end = nullptr;
        double dval = std::strtod(raw_val.c_str(), &end);
        if (!end || end != raw_val.c_str() + raw_val.size())
        {
          std::string m =
            "could not convert string to float : '" + raw_val + "'";
          return converter_.get_exception_handler().gen_exception_raise(
            "ValueError", m);
        }
        return from_double(dval, type_handler_.get_typet("float", 0));
      }
    }
  }

  // Handle float(): convert string (from symbol) to float
  else if (func_name == "float" && arg["_type"] == "Name")
  {
    const symbolt *sym = lookup_python_symbol(arg["id"]);
    if (
      sym && sym->value.is_constant() &&
      type_utils::is_string_type(sym->value.type()))
      return handle_str_symbol_to_float(sym);
    else
    {
      // Try to get the expression type directly, even if symbol lookup failed
      exprt expr = converter_.get_expr(arg);
      if (type_utils::is_string_type(expr.type()))
      {
        std::string var_name = arg["id"].get<std::string>();
        std::string m = "float() conversion may fail - variable " + var_name +
                        " may contain non-float string";

        return converter_.get_exception_handler().gen_exception_raise(
          "ValueError", m);
      }
      // Numeric variable: emit a proper typecast to avoid mislabeled IR
      typet float_t = type_handler_.get_typet("float", 0);
      if (!expr.type().is_floatbv())
        return typecast_exprt(expr, float_t);
      return expr;
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

  else if (func_name == "bool")
  {
    exprt value_expr = converter_.get_expr(arg);
    if (value_expr.is_nil())
      return value_expr;
    if (value_expr.statement() == "cpp-throw")
      return value_expr;

    if (is_complex_type(value_expr.type()))
      return complex_to_bool_expr(value_expr);

    value_expr.type() = type_handler_.get_typet(func_name, arg_size);
    return value_expr;
  }

  else if (func_name == "str")
  {
    // Try __str__ dispatch for custom objects with __str__ defined.
    exprt value_expr = converter_.get_expr(arg);
    if (!value_expr.is_nil() && value_expr.statement() != "cpp-throw")
    {
      exprt dunder_result = converter_.dispatch_unary_dunder_operator(
        "str", value_expr, converter_.get_location_from_decl(call_));
      if (!dunder_result.is_nil())
        return dunder_result;
    }
    arg_size = handle_str(arg);
  }

  typet t = type_handler_.get_typet(func_name, arg_size);
  exprt expr = converter_.get_expr(arg);

  // For float(), emit a proper typecast instead of relabeling the type.
  // Simply changing expr.type() on an integer expression creates IR where
  // the type tag says float but the operation is bitvector arithmetic,
  // causing sort mismatches in the SMT encoder.
  if (func_name == "float" && !expr.type().is_floatbv())
    return typecast_exprt(expr, t);

  if (func_name != "str")
    expr.type() = t;

  return expr;
}

std::string function_call_expr::get_object_name() const
{
  const nlohmann::json &func_json =
    (call_.contains("func") && call_["func"].is_object())
      ? call_["func"]
      : nlohmann::json::object();
  if (!func_json.contains("value") || !func_json["value"].is_object())
    return std::string();
  const auto &subelement = func_json["value"];
  const std::string node_type = node_type_of(subelement);

  std::string obj_name;
  if (node_type == "Attribute")
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
      if (subelement.contains("attr") && subelement["attr"].is_string())
        obj_name = subelement["attr"].get<std::string>();
    }
  }
  else if (node_type == "Constant" || node_type == "BinOp")
    obj_name = function_id_.get_class();
  else if (node_type == "Call")
  {
    const bool has_name_callee =
      subelement.contains("func") && subelement["func"].is_object() &&
      node_type_of(subelement["func"]) == "Name" &&
      subelement["func"].contains("id") && subelement["func"]["id"].is_string();
    if (has_name_callee)
    {
      obj_name = subelement["func"]["id"].get<std::string>();
    }
    else
    {
      // Nested call receivers (e.g. u.encode(...).decode(...)) do not have
      // a func.id field in the inner Call AST node.
      obj_name = type_handler_.get_operand_type(subelement);
    }

    if (obj_name.empty())
      obj_name = function_id_.get_class();

    if (obj_name == "super")
      obj_name = "self";
  }
  else if (node_type == "Subscript")
  {
    // Method call on a subscript result, e.g. d["key"].method().
    // We intentionally leave obj_name empty: the subscript result is a
    // temporary value, not a named symbol, so resolving obj_name to the base
    // variable (e.g. 'd' from d["k"]) would cause method handlers to operate
    // on the wrong object.  For-loop uses of dict.items() are rewritten by the
    // preprocessor into a named temp before reaching here; other methods on
    // subscript results are not yet supported.
  }
  else
  {
    // Expect a plain Name node with an "id" field. Guard against
    // missing "id" to avoid nlohmann::json::type_error on unexpected node shapes.
    if (subelement.contains("id") && subelement["id"].is_string())
      obj_name = subelement["id"].get<std::string>();
  }

  if (obj_name.empty())
    return obj_name;

  return json_utils::get_object_alias(converter_.ast(), obj_name);
}

const symbolt *
function_call_expr::get_object_list_symbol(std::string &display_name) const
{
  const auto &func_value = call_["func"]["value"];

  // Subscript case: e.g. nested[0].append(99) — resolve the inner list symbol
  // via the compile-time list_type_map rather than through a plain name lookup.
  if (func_value["_type"] == "Subscript")
  {
    const auto &base_node = func_value["value"];
    if (!base_node.contains("id"))
      return nullptr;

    std::string base_name = base_node["id"].get<std::string>();
    base_name = json_utils::get_object_alias(converter_.ast(), base_name);

    symbol_id base_sym_id = converter_.create_symbol_id();
    base_sym_id.set_object(base_name);
    const symbolt *base_sym = converter_.find_symbol(base_sym_id.to_string());
    if (!base_sym)
      return nullptr;

    const auto &slice_node = func_value["slice"];
    const typet list_type = converter_.get_type_handler().get_list_type();
    const std::string &base_id = base_sym->id.as_string();

    // Constant index: resolve directly from list_type_map.
    if (
      slice_node["_type"] == "Constant" &&
      slice_node["value"].is_number_integer())
    {
      const size_t index = slice_node["value"].get<size_t>();

      if (python_list::get_list_element_type(base_id, index) != list_type)
        return nullptr;

      const std::string inner_id =
        python_list::get_list_element_id(base_id, index);
      if (inner_id.empty())
        return nullptr;

      display_name = base_name + "[" + std::to_string(index) + "]";
      return converter_.find_symbol(inner_id);
    }

    // Non-constant index (e.g. nested[i].append(v)): delegate to the existing
    // subscript handler.  For comprehension-generated nested lists the handler
    // hits the list_type_map early-return path and yields the template inner
    // list symbol (the element produced inside the loop body) without emitting
    // any runtime instructions.
    const exprt subscript_expr = converter_.get_expr(func_value);
    if (subscript_expr.is_symbol())
    {
      const symbolt *sym =
        converter_.find_symbol(subscript_expr.identifier().as_string());
      if (sym && sym->type == list_type)
      {
        const std::string idx_str = slice_node.contains("id")
                                      ? slice_node["id"].get<std::string>()
                                      : "(expr)";
        display_name = base_name + "[" + idx_str + "]";
        return sym;
      }
    }
    return nullptr;
  }

  // Attribute case: e.g. obj.mutable_attr.append(1)
  // Resolve the attribute access via get_expr(), which already handles the
  // class-attribute fallback (instance attr not set → class-level symbol).
  if (func_value["_type"] == "Attribute")
  {
    const exprt attr_expr = converter_.get_expr(func_value);
    const typet list_type = converter_.get_type_handler().get_list_type();

    if (
      func_value.contains("value") && func_value["value"].contains("id") &&
      func_value.contains("attr"))
    {
      display_name = func_value["value"]["id"].get<std::string>() + "." +
                     func_value["attr"].get<std::string>();
    }

    if (attr_expr.is_symbol())
    {
      const symbolt *sym =
        converter_.find_symbol(attr_expr.identifier().as_string());
      if (sym && sym->type == list_type)
        return sym;
    }

    // Instance attribute: attr_expr is a member_exprt (struct field access) of
    // list pointer type. list_type is PyListObject*, so both the temp and the
    // struct member point to the same PyListObject. All list mutations through
    // the temp are visible via the original member (same pointer, same object).
    //
    // The temp symbol stores attr_expr as its value so that
    // materialize_list_symbol() can emit the declaration lazily — only in
    // handler methods, never inside discriminators (is_list_method_call, etc.).
    if (attr_expr.type() == list_type)
    {
      symbolt &tmp = converter_.create_tmp_symbol(
        call_, "$attr_list$", list_type, attr_expr);
      return &tmp;
    }

    return nullptr;
  }

  // Call case: e.g. a.setdefault(k, []).append(99)
  // receiver is a function call whose return value is a list pointer.
  // Materialize the call result into a $call_list$ temp symbol
  // so list method handlers can treat it as a named list.
  // The declaration of the temp is emitted lazily by materialize_list_symbol().
  if (func_value["_type"] == "Call")
  {
    const exprt call_expr = converter_.get_expr(func_value);
    const typet list_type = converter_.get_type_handler().get_list_type();
    if (call_expr.type() == list_type)
    {
      symbolt &tmp = converter_.create_tmp_symbol(
        call_, "$call_list$", list_type, call_expr);
      display_name = "$call_list$";
      return &tmp;
    }
    return nullptr;
  }

  // Plain name case: e.g. mylist.append(99)
  display_name = get_object_name();
  symbol_id list_symbol_id = converter_.create_symbol_id();
  list_symbol_id.set_object(display_name);
  return converter_.find_symbol(list_symbol_id.to_string());
}

// Emit the IR declaration for an instance-attribute list temp symbol.
//
// get_object_list_symbol() creates a temp symbol (named "$attr_list$...") that
// holds the member_exprt of an instance attribute list as its value.  This is
// kept as a pure lookup so that discriminators (is_list_method_call, etc.) do
// not emit IR as a side-effect.  Each list method handler calls this function
// once, just after get_object_list_symbol(), to emit the actual code_declt that
// initialises the temp pointer from the struct member.
//
// For non-instance-attribute symbols (global lists, class-level lists) the
// function is a no-op.
void function_call_expr::materialize_list_symbol(const symbolt *sym) const
{
  if (!sym || sym->value.is_nil())
    return;
  // Only emit a declaration for temp symbols produced by
  // get_object_list_symbol() from non-named receivers.  These are identified
  // by the prefixes "$attr_list$" and "$call_list$".  Regular list symbols
  // have a nil value and are declared elsewhere; this guard prevents
  // re-declaring them.
  const std::string &name = sym->name.as_string();
  if (
    name.find("$attr_list$") == std::string::npos &&
    name.find("$call_list$") == std::string::npos)
    return;
  code_declt decl(symbol_expr(*sym));
  decl.copy_to_operands(sym->value);
  decl.location() = sym->location;
  converter_.current_block->copy_to_operands(decl);
}

bool function_call_expr::is_min_max_call() const
{
  const std::string &func_name = function_id_.get_function();
  bool is_min_or_max = (func_name == "min" || func_name == "max");

  if (!is_min_or_max)
    return false;

  const auto &args = call_["args"];

  // Handle N >= 2 direct arguments: min(a, b), min(a, b, c), etc.
  if (args.size() >= 2)
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

exprt to_value_expr(const exprt &arg, const namespacet &ns)
{
  if (!arg.is_code() || !arg.is_function_call())
    return arg;

  side_effect_expr_function_callt func_call;
  func_call.function() = arg.op1();
  for (const auto &operand : to_code(arg).op2().operands())
    func_call.arguments().push_back(operand);

  const exprt &func_expr = arg.op1();
  if (func_expr.is_symbol())
  {
    const symbolt *sym = ns.lookup(to_symbol_expr(func_expr));
    if (sym)
      func_call.type() = to_code_type(sym->type).return_type();
  }
  if (func_call.type().is_nil() || func_call.type().id() == "empty")
    func_call.type() = arg.type();
  return func_call;
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

  // N >= 2 direct arguments: min(a, b, c, ...) — build a comparison chain.
  std::vector<exprt> exprs;
  exprs.reserve(args.size());
  for (const auto &arg : args)
    exprs.push_back(to_value_expr(converter_.get_expr(arg), converter_.ns));

  // Determine common promoted type across all arguments.
  typet result_type = exprs[0].type();
  for (size_t i = 1; i < exprs.size(); ++i)
  {
    const typet &t = exprs[i].type();
    if (base_type_eq(result_type, t, converter_.ns))
      continue;
    if (result_type.is_floatbv() && t.is_signedbv())
      continue; // keep float
    else if (result_type.is_signedbv() && t.is_floatbv())
      result_type = t; // promote int -> float
    else if (
      (result_type.is_signedbv() || result_type.is_unsignedbv()) &&
      (t.is_signedbv() || t.is_unsignedbv()))
    {
      unsigned wa = result_type.is_signedbv()
                      ? to_signedbv_type(result_type).get_width()
                      : to_unsignedbv_type(result_type).get_width();
      unsigned wb = t.is_signedbv() ? to_signedbv_type(t).get_width()
                                    : to_unsignedbv_type(t).get_width();
      result_type = signedbv_typet(std::max(wa, wb));
    }
    else
      throw std::runtime_error(
        func_name + "() arguments must be of comparable types: got " +
        result_type.pretty() + " and " + t.pretty());
  }

  // Cast all args to the common type.
  for (auto &e : exprs)
    if (!base_type_eq(e.type(), result_type, converter_.ns))
      e = typecast_exprt(e, result_type);

  // Fold: result = exprs[0]; for each subsequent arg update via if-expr.
  exprt result = exprs[0];
  for (size_t i = 1; i < exprs.size(); ++i)
  {
    exprt condition(comparison_op, type_handler_.get_typet("bool", 0));
    condition.copy_to_operands(exprs[i], result);
    if_exprt update(condition, exprs[i], result);
    update.type() = result_type;
    result = update;
  }

  return result;
}

exprt function_call_expr::handle_list_insert() const
{
  const auto &args = call_["args"];

  if (args.size() != 2)
    throw std::runtime_error("insert() takes exactly two arguments");

  std::string list_display_name;
  const symbolt *list_symbol = get_object_list_symbol(list_display_name);
  materialize_list_symbol(list_symbol);

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_display_name);

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
  std::string list_display_name;
  const symbolt *list_symbol = get_object_list_symbol(list_display_name);
  materialize_list_symbol(list_symbol);

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_display_name);

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

  const symbolt *list_symbol = nullptr;

  // Create temporary symbols from binary expressions (e.g., (x-y).pop()).
  if (
    call_["func"].contains("value") &&
    call_["func"]["value"].contains("_type") &&
    call_["func"]["value"]["_type"] == "BinOp")
  {
    exprt list_expr = converter_.get_expr(call_["func"]["value"]);
    if (list_expr.is_symbol())
    {
      list_symbol = converter_.symbol_table().find_symbol(
        to_symbol_expr(list_expr).get_identifier());
    }
  }

  if (!list_symbol)
  {
    std::string list_display_name;
    list_symbol = get_object_list_symbol(list_display_name);
    materialize_list_symbol(list_symbol);
    if (!list_symbol)
      throw std::runtime_error("List variable not found: " + list_display_name);
  }

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

bool function_call_expr::is_tuple_method_call() const
{
  if (call_["func"]["_type"] != "Attribute")
    return false;

  const std::string &method_name = function_id_.get_function();
  if (method_name != "count" && method_name != "index")
    return false;

  exprt receiver = converter_.get_expr(call_["func"]["value"]);
  return converter_.get_tuple_handler().is_tuple_type(receiver.type());
}

exprt function_call_expr::handle_tuple_method() const
{
  const std::string &method_name = function_id_.get_function();
  const auto &args = call_["args"];
  if (args.size() != 1)
    throw std::runtime_error(
      "tuple." + method_name + "() takes exactly one argument");

  exprt receiver = converter_.get_expr(call_["func"]["value"]);
  const struct_typet &tuple_type = to_struct_type(receiver.type());
  const auto &components = tuple_type.components();

  exprt elem = converter_.get_expr(args[0]);

  if (method_name == "count")
  {
    // sum(t.element_i == elem ? 1 : 0)
    typet result_type = int_type();
    exprt total = gen_zero(result_type);
    for (const auto &comp : components)
    {
      exprt member = member_exprt(receiver, comp.get_name(), comp.type());
      exprt eq = equality_exprt(member, elem);
      if_exprt sel(eq, gen_one(result_type), gen_zero(result_type));
      sel.type() = result_type;
      total = plus_exprt(total, sel);
      total.type() = result_type;
    }
    return total;
  }

  // method_name == "index"
  // Return the smallest k for which t.element_k == elem; assert if absent.
  if (components.empty())
    throw std::runtime_error("tuple.index() on empty tuple");

  // Build "any matched" guard so we can assert the element is present.
  typet result_type = int_type();
  exprt any_match = gen_boolean(false);
  for (const auto &comp : components)
  {
    exprt member = member_exprt(receiver, comp.get_name(), comp.type());
    exprt eq = equality_exprt(member, elem);
    any_match = or_exprt(any_match, eq);
  }
  code_assertt found_assert(any_match);
  found_assert.location() = converter_.get_location_from_decl(call_);
  found_assert.location().comment("ValueError: tuple.index(x): x not in tuple");
  converter_.add_instruction(found_assert);

  // Build chain right-to-left: result_(n-1) is index n-1, falling back to
  // earlier matches as we walk backwards. Net effect: leftmost match wins.
  size_t n = components.size();
  exprt result = from_integer(BigInt(n - 1), result_type);
  for (size_t k = n - 1; k-- > 0;)
  {
    exprt member =
      member_exprt(receiver, components[k].get_name(), components[k].type());
    exprt eq = equality_exprt(member, elem);
    if_exprt sel(eq, from_integer(BigInt(k), result_type), result);
    sel.type() = result_type;
    result = sel;
  }
  return result;
}

bool function_call_expr::is_dict_method_call() const
{
  if (call_["func"]["_type"] != "Attribute")
    return false;

  const std::string &method_name = function_id_.get_function();

  if (
    !python_dict_handler::is_value_returning_method(method_name) &&
    method_name != "update")
    return false;

  // For "pop", which exists on both list and dict, treat as dict.pop() when
  // the receiver does not resolve to a list symbol.
  if (method_name == "pop")
  {
    std::string dummy;
    const symbolt *sym = get_object_list_symbol(dummy);
    const typet list_type = type_handler_.get_list_type();
    return sym == nullptr || sym->type != list_type;
  }

  return true;
}

exprt function_call_expr::handle_dict_method() const
{
  const std::string &method_name = function_id_.get_function();

  // Resolve the dict to a symbol.
  // Named receivers look up by name;
  // inline literals (e.g. `{"a":1}.get(...)`) have none,
  // so spill into a tmp first.
  exprt dict_expr;
  std::string dict_name = get_object_name();
  if (!dict_name.empty())
  {
    symbol_id dict_symbol_id = converter_.create_symbol_id();
    dict_symbol_id.set_object(dict_name);
    const symbolt *dict_symbol =
      converter_.find_symbol(dict_symbol_id.to_string());
    if (!dict_symbol)
      throw std::runtime_error("Dictionary variable not found: " + dict_name);
    dict_expr = symbol_expr(*dict_symbol);
  }
  else
  {
    exprt literal = converter_.get_expr(call_["func"]["value"]);
    symbolt &tmp = converter_.create_tmp_symbol(
      call_, "$dict_lit$", literal.type(), exprt());
    converter_.add_instruction(code_declt(symbol_expr(tmp)));
    converter_.add_instruction(code_assignt(symbol_expr(tmp), literal));
    dict_expr = symbol_expr(tmp);
  }

  if (method_name == "get")
    return converter_.get_dict_handler()->handle_dict_get(dict_expr, call_);

  if (method_name == "setdefault")
    return converter_.get_dict_handler()->handle_dict_setdefault(
      dict_expr, call_);

  if (method_name == "update")
    return converter_.get_dict_handler()->handle_dict_update(dict_expr, call_);

  if (method_name == "pop")
    return converter_.get_dict_handler()->handle_dict_pop(dict_expr, call_);

  if (method_name == "popitem")
    return converter_.get_dict_handler()->handle_dict_popitem(dict_expr, call_);

  if (method_name == "copy")
    return converter_.get_dict_handler()->handle_dict_copy(dict_expr, call_);

  throw std::runtime_error("Unsupported dict method: " + method_name);
}

bool function_call_expr::is_dict_class_method_call() const
{
  if (call_["func"]["_type"] != "Attribute")
    return false;
  const auto &value = call_["func"]["value"];
  if (!value.contains("_type") || value["_type"] != "Name")
    return false;
  if (value["id"] != "dict")
    return false;

  const std::string &method_name = function_id_.get_function();
  return method_name == "fromkeys";
}

exprt function_call_expr::handle_list_copy() const
{
  const auto &args = call_["args"];

  if (!args.empty())
    throw std::runtime_error("copy() takes no arguments");

  std::string list_display_name;
  const symbolt *list_symbol = get_object_list_symbol(list_display_name);
  materialize_list_symbol(list_symbol);

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_display_name);

  // Delegate to python_list to build the copy operation
  python_list list_helper(converter_, call_);
  return list_helper.build_copy_list_call(*list_symbol, call_);
}

exprt function_call_expr::handle_list_remove() const
{
  const auto &args = call_["args"];

  if (args.size() != 1)
    throw std::runtime_error("remove() takes exactly one argument");

  std::string list_display_name;
  const symbolt *list_symbol = get_object_list_symbol(list_display_name);
  materialize_list_symbol(list_symbol);

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_display_name);

  exprt value_to_remove = converter_.get_expr(args[0]);

  python_list list_helper(converter_, call_);
  exprt result =
    list_helper.build_remove_list_call(*list_symbol, call_, value_to_remove);

  return result;
}

exprt function_call_expr::handle_list_sort() const
{
  const auto &args = call_["args"];
  if (!args.empty())
    throw std::runtime_error(
      "sort() positional arguments are not supported; "
      "use sort() with no arguments");

  // sort() supports a `reverse=` keyword. `key=` is not handled yet; reject
  // explicitly rather than dropping it silently.
  bool reverse = false;
  if (call_.contains("keywords"))
  {
    for (const auto &kw : call_["keywords"])
    {
      const std::string name = kw.value("arg", "");
      if (name == "reverse")
      {
        exprt v = converter_.get_expr(kw["value"]);
        if (!v.is_constant())
          throw std::runtime_error(
            "sort(reverse=...) requires a constant boolean");
        reverse = v.is_true();
      }
      else
        throw std::runtime_error(
          "sort() keyword argument '" + name +
          "' is not supported (only reverse=)");
    }
  }

  std::string list_display_name;
  const symbolt *list_symbol = get_object_list_symbol(list_display_name);
  materialize_list_symbol(list_symbol);

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_display_name);

  const std::string &list_id = list_symbol->id.as_string();

  // ── Determine type_flag and float_type_id ─────────────────────────────────
  //
  // type_flag:
  //   0 = all-integer          → int64_t comparison    (SMT-fast, no FP)
  //   1 = all-float            → *(double*) bit-read
  //   2 = string/lexicographic → memcmp
  //   3 = mixed int + float    → per-element dispatch via float_type_id

  int type_flag = 0;
  size_t float_type_id = 0;
  python_list::get_list_type_flags(
    list_id, converter_.get_type_handler(), type_flag, float_type_id);

  // ── Locate the C model function ────────────────────────────────────────────
  const symbolt *sort_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_sort");
  if (!sort_func)
    throw std::runtime_error(
      "__ESBMC_list_sort function not found in symbol table");

  // ── Emit the call: __ESBMC_list_sort(list, type_flag, float_type_id) ──────
  code_function_callt sort_call;
  sort_call.function() = symbol_expr(*sort_func);
  sort_call.arguments().push_back(symbol_expr(*list_symbol));
  sort_call.arguments().push_back(from_integer(type_flag, int_type()));
  sort_call.arguments().push_back(
    from_integer(float_type_id, unsignedbv_typet(config.ansi_c.address_width)));
  sort_call.type() = empty_typet();
  sort_call.location() = converter_.get_location_from_decl(call_);

  // For reverse=True, sort ascending then reverse in place via the existing
  // __ESBMC_list_reverse model. Wrap both calls in a code_blockt.
  if (!reverse)
    return sort_call;

  const symbolt *reverse_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_reverse");
  if (!reverse_func)
    throw std::runtime_error(
      "__ESBMC_list_reverse function not found in symbol table");

  code_function_callt reverse_call;
  reverse_call.function() = symbol_expr(*reverse_func);
  reverse_call.arguments().push_back(symbol_expr(*list_symbol));
  reverse_call.type() = empty_typet();
  reverse_call.location() = converter_.get_location_from_decl(call_);

  python_list::reverse_type_info(list_id);

  code_blockt block;
  block.copy_to_operands(sort_call);
  block.copy_to_operands(reverse_call);
  return block;
}

exprt function_call_expr::handle_list_reverse() const
{
  const auto &args = call_["args"];

  if (!args.empty())
    throw std::runtime_error("reverse() takes no arguments");

  std::string list_display_name;
  const symbolt *list_symbol = get_object_list_symbol(list_display_name);
  materialize_list_symbol(list_symbol);

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_display_name);

  // Locate the C model function __ESBMC_list_reverse
  const symbolt *reverse_func =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_reverse");
  assert(reverse_func);

  // Emit: __ESBMC_list_reverse(list)
  code_function_callt reverse_call;
  reverse_call.function() = symbol_expr(*reverse_func);
  reverse_call.arguments().push_back(symbol_expr(*list_symbol));
  reverse_call.type() = empty_typet();
  reverse_call.location() = converter_.get_location_from_decl(call_);

  // Reverse the compile-time type-info vector to mirror the runtime
  // reordering, so that subsequent index-based type lookups remain valid.
  python_list::reverse_type_info(list_symbol->id.as_string());

  return reverse_call;
}

bool function_call_expr::is_list_method_call() const
{
  if (call_["func"]["_type"] != "Attribute")
    return false;

  const std::string &method_name = function_id_.get_function();

  if (
    method_name != "append" && method_name != "pop" &&
    method_name != "insert" && method_name != "remove" &&
    method_name != "clear" && method_name != "extend" &&
    method_name != "copy" && method_name != "sort" && method_name != "reverse")
    return false;

  // "pop" is shared between list and dict. Disambiguate using the actual
  // symbol type: only treat as list.pop() when the receiver resolves to a
  // symbol whose type is the list type.
  if (method_name == "pop")
  {
    // A BinOp receiver (e.g., (s1 - s2).pop()) is always a set/list: dicts
    // do not support arithmetic operators. handle_list_pop() already handles
    // this case, so route it here before the symbol-type check.
    if (
      call_["func"].contains("value") &&
      call_["func"]["value"].contains("_type") &&
      call_["func"]["value"]["_type"] == "BinOp")
      return true;

    std::string dummy;
    const symbolt *sym = get_object_list_symbol(dummy);
    const typet list_type = type_handler_.get_list_type();
    return sym != nullptr && sym->type == list_type;
  }

  // "copy" is shared between list and dict. Treat as list.copy() only when
  // the receiver resolves to a list symbol; otherwise let dispatch fall
  // through to handle_dict_method().
  if (method_name == "copy")
  {
    // BinOp receivers (e.g. (s1 - s2).copy()) are list-like, since dicts do
    // not support arithmetic operators.
    if (
      call_["func"].contains("value") &&
      call_["func"]["value"].contains("_type") &&
      call_["func"]["value"]["_type"] == "BinOp")
      return true;

    std::string dummy;
    const symbolt *sym = get_object_list_symbol(dummy);
    const typet list_type = type_handler_.get_list_type();
    return sym != nullptr && sym->type == list_type;
  }

  return true;
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
  if (method_name == "sort")
    return handle_list_sort();
  if (method_name == "reverse")
    return handle_list_reverse();
  // Add other methods as needed

  throw std::runtime_error("Unsupported list method: " + method_name);
}

exprt function_call_expr::handle_list_append() const
{
  const auto &args = call_["args"];

  if (args.size() != 1)
    throw std::runtime_error("append() takes exactly one argument");

  std::string list_display_name;
  const symbolt *list_symbol = get_object_list_symbol(list_display_name);
  materialize_list_symbol(list_symbol);

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_display_name);

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

  std::string list_display_name;
  const symbolt *list_symbol = get_object_list_symbol(list_display_name);
  materialize_list_symbol(list_symbol);

  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_display_name);

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
  // Materialize each argument as code so arithmetic checks and function-call
  // side effects are preserved even though print itself has no runtime output.
  const auto &args = call_["args"];
  for (const auto &arg_node : args)
  {
    // Direct call arguments (print(f(...))) are currently lowered through
    // the regular expression flow and may trigger invalid cast paths when
    // re-materialized as expression statements here.
    // Keep them non-materialized for now and only materialize non-call
    // expressions such as arithmetic operators (e.g., print(a + b)).
    if (arg_node.contains("_type") && arg_node["_type"] == "Call")
      continue;

    exprt arg_expr = converter_.get_expr(arg_node);
    if (arg_expr.is_nil())
      throw std::runtime_error(
        "Failed to convert print() argument to expression");

    // Trivial values have no side effects or checks to materialize.
    if (arg_expr.is_constant() || arg_expr.id() == "symbol")
      continue;

    codet arg_code = converter_.convert_expression_to_code(arg_expr);
    converter_.current_block->copy_to_operands(arg_code);
  }

  // print() has no meaningful return value.
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

// Check if an AST node is a known-empty literal (falsy in Python).
// Needed because ESBMC's IR represents empty containers as non-NULL
// pointers/structs, making them appear truthy at the IR level.
static bool is_empty_literal(const nlohmann::json &node)
{
  const std::string &type = node["_type"];
  if (type == "List" || type == "Tuple" || type == "Set")
    return node.contains("elts") && node["elts"].empty();
  if (type == "Dict")
    return node.contains("keys") && node["keys"].empty();
  if (type == "Constant" && node.contains("value") && node["value"].is_string())
    return node["value"].get<std::string>().empty();
  return false;
}

exprt function_call_expr::compute_element_truthiness(const exprt &element) const
{
  if (element.type() == none_type())
    return gen_boolean(false);

  if (element.type().is_bool())
    return element;

  if (
    element.type().id() == "signedbv" || element.type().id() == "unsignedbv" ||
    element.type().id() == "floatbv" || element.type().is_pointer())
    return not_exprt(equality_exprt(element, gen_zero(element.type())));

  if (is_complex_type(element.type()))
    return complex_to_bool_expr(element);

  // For other types, assume truthy (conservative)
  return gen_boolean(true);
}

bool function_call_expr::is_any_call() const
{
  const std::string &func_name = function_id_.get_function();
  return func_name == "any";
}

exprt function_call_expr::handle_any()
{
  const auto keywords =
    call_.contains("keywords") ? call_["keywords"] : nlohmann::json::array();
  if (!keywords.empty())
    throw std::runtime_error("any() takes no keyword arguments");

  const auto &args = call_["args"];

  if (args.empty())
    throw std::runtime_error("any() expected at least 1 argument, got 0");

  if (args.size() > 1)
    throw std::runtime_error("any() takes at most 1 argument");

  const auto &arg = args[0];
  const std::string &arg_type = arg["_type"];

  if (arg_type == "List" || arg_type == "Tuple" || arg_type == "Set")
    return reduce_iterable_literal_truthiness(arg, ReduceOp::Any);

  exprt arg_expr = converter_.get_expr(arg);
  if (converter_.get_tuple_handler().is_tuple_type(arg_expr.type()))
    return reduce_tuple_expr_truthiness(arg_expr, ReduceOp::Any);

  // Set / frozenset variables share the PyListObject* representation, so
  // forward to the list-backed any() model.
  if (arg_expr.type() == converter_.get_type_handler().get_list_type())
    return handle_general_function_call();

  throw std::runtime_error(
    "any() currently only supports list/tuple/set literals or list, set, or "
    "tuple variables");
}

bool function_call_expr::is_all_call() const
{
  const std::string &func_name = function_id_.get_function();
  return func_name == "all";
}

exprt function_call_expr::handle_all()
{
  const auto keywords =
    call_.contains("keywords") ? call_["keywords"] : nlohmann::json::array();
  if (!keywords.empty())
    throw std::runtime_error("all() takes no keyword arguments");

  const auto &args = call_["args"];

  if (args.empty())
    throw std::runtime_error("all() expected at least 1 argument, got 0");

  if (args.size() > 1)
    throw std::runtime_error(
      "all() takes at most 1 argument, got " + std::to_string(args.size()));

  const auto &arg = args[0];
  const std::string &arg_type = arg["_type"];

  if (arg_type == "List" || arg_type == "Tuple" || arg_type == "Set")
    return reduce_iterable_literal_truthiness(arg, ReduceOp::All);

  // Non-literal argument: forward to the Python operational model only
  // when the value is actually a list (pointer to PyListObj). Tuple
  // values are evaluated by combining the truthiness of each struct
  // member; anything else gets a clear error rather than being silently
  // passed to __ESBMC_list_size, which would dereference a non-list
  // pointer (issue #4295).
  exprt arg_expr = converter_.get_expr(arg);
  if (converter_.get_tuple_handler().is_tuple_type(arg_expr.type()))
    return reduce_tuple_expr_truthiness(arg_expr, ReduceOp::All);

  // Sets share the PyListObject* representation, so the list-backed all()
  // model handles set variables transparently.
  if (arg_expr.type() == converter_.get_type_handler().get_list_type())
    return handle_general_function_call();

  throw std::runtime_error(
    "all() currently only supports list/tuple/set literals, list, set, or "
    "tuple variables");
}

exprt function_call_expr::reduce_iterable_literal_truthiness(
  const nlohmann::json &iterable_arg,
  ReduceOp op) const
{
  const auto &elts = iterable_arg["elts"];

  if (elts.empty())
    return gen_boolean(op == ReduceOp::All);

  exprt result;
  bool first = true;

  for (const auto &elt : elts)
  {
    exprt is_truthy;

    if (is_empty_literal(elt))
    {
      is_truthy = gen_boolean(false);
    }
    else
    {
      exprt element = converter_.get_expr(elt);
      is_truthy = compute_element_truthiness(element);
    }

    if (first)
    {
      result = is_truthy;
      first = false;
      continue;
    }

    result = (op == ReduceOp::Any) ? exprt(or_exprt(result, is_truthy))
                                   : exprt(and_exprt(result, is_truthy));
  }

  return result;
}

exprt function_call_expr::reduce_tuple_expr_truthiness(
  const exprt &tuple_expr,
  ReduceOp op) const
{
  const struct_typet &tuple_type = to_struct_type(tuple_expr.type());
  const auto &components = tuple_type.components();

  if (components.empty())
    return gen_boolean(op == ReduceOp::All);

  exprt result;
  bool first = true;
  for (const auto &component : components)
  {
    member_exprt member(tuple_expr, component.get_name(), component.type());
    exprt is_truthy = compute_element_truthiness(member);

    if (first)
    {
      result = is_truthy;
      first = false;
      continue;
    }

    result = (op == ReduceOp::Any) ? exprt(or_exprt(result, is_truthy))
                                   : exprt(and_exprt(result, is_truthy));
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

    // All function
    {[this]() { return is_all_call(); },
     [this]() { return handle_all(); },
     "all()"},

    // int.to_bytes()
    {[this]() {
       if (call_["func"]["_type"] != "Attribute")
         return false;
       if (function_id_.get_function() != "to_bytes")
         return false;

       const auto &obj = call_["func"]["value"];
       return (obj["_type"] == "Name" && obj["id"] == "int") ||
              (obj["_type"] == "Name" &&
               type_handler_.get_var_type(obj["id"]) == "int");
     },
     [this]() { return handle_int_to_bytes(); },
     "int.to_bytes()"},

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

    // __iter__ on builtin iterables (range, list, tuple, str, set, etc.)
    // Returns the object itself: we model iteration via index-based while
    // loops, so the iterable is the iterator.
    {[this]() {
       if (call_["func"]["_type"] != "Attribute")
         return false;
       if (function_id_.get_function() != "__iter__")
         return false;
       std::string obj_type = type_handler_.get_var_type(get_object_name());
       return type_utils::is_builtin_type(obj_type);
     },
     [this]() { return converter_.get_expr(call_["func"]["value"]); },
     "__iter__ on builtin iterables"},

    // Tuple methods (count, index) — matched before list/dict so a tuple
    // receiver doesn't fall through to either.
    {[this]() { return is_tuple_method_call(); },
     [this]() { return handle_tuple_method(); },
     "tuple methods"},

    // List methods
    {[this]() { return is_list_method_call(); },
     [this]() { return handle_list_method(); },
     "list methods"},

    // Dict class methods (dict.fromkeys), matched before instance-method dispatch
    // The receiver is the class name, not a dict symbol.
    {[this]() { return is_dict_class_method_call(); },
     [this]() {
       return converter_.get_dict_handler()->handle_dict_fromkeys(call_);
     },
     "dict class methods"},

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
         if (is_cpp_throw_expr(arg_expr))
           return arg_expr;
         if (is_complex_type(arg_expr.type()))
           return complex_utils::raise_math_real_type_error_expr(converter_);
         exprt isnan_expr("isnan", bool_typet());
         isnan_expr.copy_to_operands(arg_expr);
         return isnan_expr;
       }
       else // __ESBMC_isinf
       {
         if (args.size() != 1)
           throw std::runtime_error("isinf() expects exactly 1 argument");

         exprt arg_expr = converter_.get_expr(args[0]);
         if (is_cpp_throw_expr(arg_expr))
           return arg_expr;
         if (is_complex_type(arg_expr.type()))
           return complex_utils::raise_math_real_type_error_expr(converter_);
         exprt isinf_expr("isinf", bool_typet());
         isinf_expr.copy_to_operands(arg_expr);
         return isinf_expr;
       }
     },
     "isnan/isinf"},

    // cmath.log / cmath.log10: lower directly to complex-safe IR to avoid
    // backend typing mismatches from model-level dispatch.
    {[this]() {
       if (!(call_.contains("func") && call_["func"].contains("_type") &&
             call_["func"]["_type"] == "Attribute"))
         return false;
       const std::string caller = get_object_name();
       if (caller != "cmath")
         return false;
       const std::string &func_name = function_id_.get_function();
       return func_name == "log" || func_name == "log10";
     },
     [this]() -> exprt {
       const std::string &raw_func_name = function_id_.get_function();
       std::string func_name = raw_func_name;
       if (
         raw_func_name.size() > 8 &&
         raw_func_name.compare(0, 8, "__ESBMC_") == 0)
       {
         func_name = raw_func_name.substr(8);
       }
       const auto &args = call_["args"];
       const auto &keywords = call_.contains("keywords")
                                ? call_["keywords"]
                                : nlohmann::json::array();

       // Budget guard: when the argument is structurally expensive and the
       // model symbol exists, delegate to avoid solver blow-up.
       if (args.size() == 1 && !cmath_lowering_policy::within_budget(args[0]))
       {
         const symbolt *model_sym =
           cached_find_symbol(function_id_.to_string());
         if (model_sym)
         {
           exprt z = converter_.get_expr(args[0]);
           if (!is_cpp_throw_expr(z))
             z = promote_to_complex(z);
           if (!is_cpp_throw_expr(z))
           {
             side_effect_expr_function_callt model_call;
             model_call.function() = symbol_expr(*model_sym);
             model_call.arguments() = {z};
             model_call.type() = to_code_type(model_sym->type).return_type();
             model_call.location() = converter_.get_location_from_decl(call_);
             return model_call;
           }
         }
       }

       return converter_.get_complex_handler().handle_cmath_log(
         func_name, call_, args, keywords);
     },
     "cmath log/log10"},

    // cmath inverse functions: use a fast path only on pure-imaginary inputs
    // and delegate all other cases to the Python cmath model implementation.
    {[this]() {
       if (!(call_.contains("func") && call_["func"].contains("_type") &&
             call_["func"]["_type"] == "Attribute"))
         return false;
       const std::string caller = get_object_name();
       if (caller != "cmath")
         return false;
       const std::string &func_name = function_id_.get_function();
       return (
         func_name == "asin" || func_name == "atan" || func_name == "asinh" ||
         func_name == "atanh");
     },
     [this]() -> exprt {
       const std::string &raw_func_name = function_id_.get_function();
       const std::string func_name = raw_func_name.rfind("__ESBMC_", 0) == 0
                                       ? raw_func_name.substr(8)
                                       : raw_func_name;
       const auto &args = call_["args"];
       const auto keywords = call_.contains("keywords")
                               ? call_["keywords"]
                               : nlohmann::json::array();

       auto raise_type_error = [this](const std::string &msg) -> exprt {
         return converter_.get_exception_handler().gen_exception_raise(
           "TypeError", msg);
       };

       if (!keywords.empty())
         return raise_type_error(
           "cmath." + func_name + "() takes no keyword arguments");
       if (args.size() != 1)
         return raise_type_error(func_name + "() takes exactly 1 argument");

       exprt z = converter_.get_expr(args[0]);
       if (is_cpp_throw_expr(z))
         return z;
       z = promote_to_complex(z);
       if (is_cpp_throw_expr(z))
         return z;

       const symbolt *model_symbol =
         cached_find_symbol(function_id_.to_string());
       if (model_symbol == nullptr)
       {
         return converter_.get_exception_handler().gen_exception_raise(
           "AttributeError",
           "module 'cmath' has no attribute '" + func_name + "'");
       }

       side_effect_expr_function_callt model_call;
       model_call.function() = symbol_expr(*model_symbol);
       model_call.arguments() = {z};
       model_call.type() = to_code_type(model_symbol->type).return_type();
       model_call.location() = converter_.get_location_from_decl(call_);

       exprt zr = member_exprt(z, "real", double_type());
       exprt zi = member_exprt(z, "imag", double_type());

       exprt imag_result;
       if (func_name == "asin")
         imag_result = converter_.get_math_handler().handle_asinh(zi, call_);
       else if (func_name == "atan")
         imag_result = converter_.get_math_handler().handle_atanh(zi, call_);
       else if (func_name == "asinh")
         imag_result = converter_.get_math_handler().handle_asin(zi, call_);
       else
         imag_result = converter_.get_math_handler().handle_atan(zi, call_);

       if (is_cpp_throw_expr(imag_result))
         return imag_result;

       exprt fast_path =
         make_complex(from_double(0.0, double_type()), imag_result);
       exprt zero = from_double(0.0, double_type());
       exprt fast_guard = equality_exprt(zr, zero);

       // For atan(i*y) and asinh(i*y), the pure-imag shortcut only matches
       // the principal branch safely within the unit interval.
       if (func_name == "atan" || func_name == "asinh")
       {
         exprt abs_zi = converter_.get_math_handler().handle_fabs(zi, call_);
         if (is_cpp_throw_expr(abs_zi))
           return abs_zi;

         exprt one = from_double(1.0, double_type());
         exprt imag_guard =
           func_name == "atan"
             ? static_cast<exprt>(binary_relation_exprt(abs_zi, "<", one))
             : static_cast<exprt>(binary_relation_exprt(abs_zi, "<=", one));
         fast_guard = and_exprt(fast_guard, imag_guard);
       }

       return if_exprt(fast_guard, fast_path, model_call);
     },
     "cmath inverse pure-imag fast path"},

    // Math module functions (sin, cos, sqrt, exp, log, etc.)
    {[this]() {
       const std::string &func_name = function_id_.get_function();
       std::string caller;
       if (
         call_.contains("func") && call_["func"].is_object() &&
         call_["func"].contains("_type") &&
         call_["func"]["_type"] == "Attribute")
       {
         caller = get_object_name();
       }
       return converter_.get_math_handler().is_math_dispatch_target_cached(
         caller, func_name);
     },
     [this]() -> exprt {
       const std::string &raw_func_name = function_id_.get_function();
       const std::string func_name = raw_func_name.rfind("__ESBMC_", 0) == 0
                                       ? raw_func_name.substr(8)
                                       : raw_func_name;
       const auto &args = call_["args"];
       auto raise_math_real_type_error = [this]() -> exprt {
         return complex_utils::raise_math_real_type_error_expr(converter_);
       };
       auto raise_math_int_type_error = [this]() -> exprt {
         return complex_utils::raise_math_int_type_error_expr(converter_);
       };
       auto has_complex_arg = [](const exprt &arg_expr) -> bool {
         return is_complex_type(arg_expr.type());
       };
       const auto call_has_complex = [&]() -> bool {
         return math_guard_utils::call_has_complex_in_args_or_keywords(
           call_,
           converter_,
           type_handler_,
           converter_.current_function_name());
       };
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
       auto validate_real_arg =
         [&](const exprt &arg_expr) -> std::optional<exprt> {
         if (is_cpp_throw_expr(arg_expr))
           return arg_expr;
         if (has_complex_arg(arg_expr))
           return raise_math_real_type_error();
         return std::nullopt;
       };
       auto validate_real_args =
         [&](
           const exprt &lhs_expr,
           const exprt &rhs_expr) -> std::optional<exprt> {
         if (is_cpp_throw_expr(lhs_expr))
           return lhs_expr;
         if (is_cpp_throw_expr(rhs_expr))
           return rhs_expr;
         if (has_complex_arg(lhs_expr) || has_complex_arg(rhs_expr))
           return raise_math_real_type_error();
         return std::nullopt;
       };

       // Fast dispatch path for math functions that do not need extra
       // domain guards in this layer (e.g., sqrt/log/acos stay in slow path).
       if (args.size() == 1)
       {
         exprt arg_expr = require_one_arg();
         if (std::optional<exprt> type_error = validate_real_arg(arg_expr);
             type_error.has_value())

           return *type_error;

         exprt dispatched =
           converter_.get_math_handler().handle(func_name, arg_expr, call_);
         if (!dispatched.is_nil())
           return dispatched;
       }

       if (args.size() == 2)
       {
         auto [lhs_expr, rhs_expr] = require_two_args();
         if (std::optional<exprt> type_error =
               validate_real_args(lhs_expr, rhs_expr);
             type_error.has_value())

           return *type_error;

         exprt dispatched = converter_.get_math_handler().handle(
           func_name, lhs_expr, rhs_expr, call_);
         if (!dispatched.is_nil())
           return dispatched;
       }

       // Enforce canonical arity/error behavior for handled names even when
       // args count is wrong, without duplicating per-function dispatch here.
       if (converter_.get_math_handler().is_unary_dispatch_function(func_name))
       {
         exprt arg_expr = require_one_arg();
         if (std::optional<exprt> type_error = validate_real_arg(arg_expr);
             type_error.has_value())

           return *type_error;
         return converter_.get_math_handler().handle(
           func_name, arg_expr, call_);
       }
       if (converter_.get_math_handler().is_binary_dispatch_function(func_name))
       {
         auto [lhs_expr, rhs_expr] = require_two_args();
         if (std::optional<exprt> type_error =
               validate_real_args(lhs_expr, rhs_expr);
             type_error.has_value())

           return *type_error;
         return converter_.get_math_handler().handle(
           func_name, lhs_expr, rhs_expr, call_);
       }

       if (func_name == "sin")
       {
         exprt arg_expr = require_one_arg();
         if (std::optional<exprt> type_error = validate_real_arg(arg_expr);
             type_error.has_value())

           return *type_error;
         return converter_.get_math_handler().handle_sin(arg_expr, call_);
       }
       else if (func_name == "cos")
       {
         exprt arg_expr = require_one_arg();
         if (std::optional<exprt> type_error = validate_real_arg(arg_expr);
             type_error.has_value())

           return *type_error;
         return converter_.get_math_handler().handle_cos(arg_expr, call_);
       }
       else if (func_name == "exp")
       {
         exprt arg_expr = require_one_arg();
         if (std::optional<exprt> type_error = validate_real_arg(arg_expr);
             type_error.has_value())

           return *type_error;
         return converter_.get_math_handler().handle_exp(arg_expr, call_);
       }
       else if (func_name == "sqrt")
       {
         exprt arg_expr = require_one_arg();
         if (std::optional<exprt> type_error = validate_real_arg(arg_expr);
             type_error.has_value())

           return *type_error;
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
       else if (func_name == "log")
       {
         exprt arg_expr = require_one_arg();
         if (std::optional<exprt> type_error = validate_real_arg(arg_expr);
             type_error.has_value())

           return *type_error;
         // Domain check for log: operand must be > 0
         exprt fp_operand = arg_expr;
         if (!arg_expr.type().is_floatbv())
         {
           fp_operand = exprt("typecast", type_handler_.get_typet("float", 0));
           fp_operand.copy_to_operands(arg_expr);
         }
         exprt zero = gen_zero(fp_operand.type());
         exprt domain_check = exprt("<=", type_handler_.get_typet("bool", 0));
         domain_check.copy_to_operands(fp_operand, zero);
         exprt raise_expr =
           converter_.get_exception_handler().gen_exception_raise(
             "ValueError", "math domain error");
         locationt loc = converter_.get_location_from_decl(call_);
         raise_expr.location() = loc;
         raise_expr.location().user_provided(true);
         code_expressiont raise_code(raise_expr);
         raise_code.location() = loc;
         code_ifthenelset guard;
         guard.cond() = domain_check;
         guard.then_case() = raise_code;
         guard.location() = loc;
         converter_.current_block->copy_to_operands(guard);
         return converter_.get_math_handler().handle_log(arg_expr, call_);
       }
       else if (func_name == "acos")
       {
         exprt arg_expr = require_one_arg();
         if (std::optional<exprt> type_error = validate_real_arg(arg_expr);
             type_error.has_value())

           return *type_error;
         // Domain check for acos: operand must be in [-1.0, 1.0]
         exprt double_operand = arg_expr;
         if (!arg_expr.type().is_floatbv())
         {
           double_operand =
             exprt("typecast", type_handler_.get_typet("float", 0));
           double_operand.copy_to_operands(arg_expr);
         }

         typet float_type = type_handler_.get_typet("float", 0);
         typet bool_t = type_handler_.get_typet("bool", 0);

         exprt pos_one = gen_one(float_type);
         exprt neg_one("unary-", float_type);
         neg_one.copy_to_operands(pos_one);

         exprt lt_neg = exprt("<", bool_t);
         lt_neg.copy_to_operands(double_operand, neg_one);
         exprt gt_pos = exprt(">", bool_t);
         gt_pos.copy_to_operands(double_operand, pos_one);
         exprt domain_check = exprt("or", bool_t);
         domain_check.copy_to_operands(lt_neg, gt_pos);

         exprt raise_expr =
           converter_.get_exception_handler().gen_exception_raise(
             "ValueError", "math domain error");
         locationt loc = converter_.get_location_from_decl(call_);
         raise_expr.location() = loc;
         raise_expr.location().user_provided(true);

         code_expressiont raise_code(raise_expr);
         raise_code.location() = loc;

         code_ifthenelset guard;
         guard.cond() = domain_check;
         guard.then_case() = raise_code;
         guard.location() = loc;

         converter_.current_block->copy_to_operands(guard);

         return converter_.get_math_handler().handle_acos(arg_expr, call_);
       }
       else if (
         math_guard_utils::math_guard_real_general_functions().count(
           func_name) != 0)
       {
         exprt arg_expr = require_one_arg();
         if (std::optional<exprt> type_error = validate_real_arg(arg_expr);
             type_error.has_value())

           return *type_error;
         return handle_general_function_call();
       }
       else if (
         math_guard_utils::math_guard_int_general_functions().count(
           func_name) != 0)
       {
         exprt throw_expr;
         if (math_guard_utils::call_first_cpp_throw_in_args_or_keywords(
               call_, converter_, throw_expr))
           return throw_expr;
         if (call_has_complex())
           return raise_math_int_type_error();
         return handle_general_function_call();
       }
       else if (
         math_guard_utils::math_guard_real_general_twoarg_functions().count(
           func_name) != 0)
       {
         exprt throw_expr;
         if (math_guard_utils::call_first_cpp_throw_in_args_or_keywords(
               call_, converter_, throw_expr))
           return throw_expr;
         if (call_has_complex())
           return raise_math_real_type_error();
         return handle_general_function_call();
       }
       else if (func_name == "dist")
       {
         exprt throw_expr;
         if (math_guard_utils::call_first_cpp_throw_in_args_or_keywords(
               call_, converter_, throw_expr))
           return throw_expr;
         if (call_has_complex())
           return raise_math_real_type_error();
         auto [lhs_expr, rhs_expr] = require_two_args();
         if (std::optional<exprt> type_error =
               validate_real_args(lhs_expr, rhs_expr);
             type_error.has_value())

           return *type_error;
         // Native handler for tuple arguments; lists use the model
         if (lhs_expr.type().is_struct() && rhs_expr.type().is_struct())
         {
           // If either argument is a constant struct (tuple literal), store it
           // in a temporary local variable so that the GOTO IR has a proper
           // symbol whose address the solver can track.
           auto materialize = [&](exprt &arg) {
             if (arg.is_constant())
             {
               symbolt &tmp = converter_.create_tmp_symbol(
                 call_, "$dist_arg$", arg.type(), arg);
               code_declt decl(symbol_expr(tmp));
               decl.location() = converter_.get_location_from_decl(call_);
               converter_.current_block->copy_to_operands(decl);
               arg = symbol_expr(tmp);
             }
           };
           materialize(lhs_expr);
           materialize(rhs_expr);
           return converter_.get_math_handler().handle_dist(
             lhs_expr, rhs_expr, call_);
         }
         return handle_general_function_call();
       }
       else if (func_name == "fsum")
       {
         exprt throw_expr;
         if (math_guard_utils::call_first_cpp_throw_in_args_or_keywords(
               call_, converter_, throw_expr))
           return throw_expr;
         if (call_has_complex())
           return raise_math_real_type_error();
         return handle_general_function_call();
       }
       else if (func_name == "sumprod" || func_name == "prod")
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

    // round() builtin function
    {[this]() {
       const std::string &func_name = function_id_.get_function();
       return func_name == "round" && function_id_.get_prefix() == "py:";
     },
     [this]() {
       if (call_["args"].empty())
         return converter_.get_exception_handler().gen_exception_raise(
           "TypeError", "round() missing required argument");
       auto arg = call_["args"][0];
       return handle_round(arg);
     },
     "round() builtin"},

    // type() built-in
    {[this]() { return function_id_.get_function() == "type"; },
     [this]() { return handle_type_call(); },
     "type()"},

    // repr() built-in — handle complex, delegate rest to general call
    {[this]() {
       return function_id_.get_function() == "repr" && !call_["args"].empty();
     },
     [this]() {
       const auto &arg = call_["args"][0];
       double real_val = 0.0, imag_val = 0.0;
       if (try_extract_complex_parts_from_json(
             arg,
             converter_.ast(),
             converter_.current_function_name(),
             real_val,
             imag_val))
       {
         return converter_.get_string_builder().build_string_literal(
           format_complex_string(real_val, imag_val));
       }
       exprt value_expr = converter_.get_expr(arg);
       if (
         !value_expr.is_nil() && value_expr.statement() != "cpp-throw" &&
         is_complex_type(value_expr.type()))
         return handle_complex_to_str();
       return handle_general_function_call();
     },
     "repr()"},

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

  // Handle single-argument min/max/sum/sorted by dispatching to typed builtins
  const std::string &func_name = function_id_.get_function();
  std::string actual_func_name = func_name;

  // Fast-path: sorted() over a concrete int list can be materialized directly
  // in the frontend, avoiding expensive runtime list sorting/equality paths.
  // Honour reverse=<constant bool>; bail to the model for any other keyword.
  if (func_name == "sorted" && call_["args"].size() == 1)
  {
    bool fast_path_reverse = false;
    bool fast_path_ok = true;
    if (call_.contains("keywords"))
    {
      for (const auto &kw : call_["keywords"])
      {
        if (kw.value("arg", "") != "reverse")
        {
          fast_path_ok = false;
          break;
        }
        exprt v = converter_.get_expr(kw["value"]);
        if (!v.is_constant())
        {
          fast_path_ok = false;
          break;
        }
        fast_path_reverse = v.is_true();
      }
    }

    if (fast_path_ok)
    {
      exprt list_arg = converter_.get_expr(call_["args"][0]);
      if (list_arg.is_symbol())
      {
        const std::string list_id = list_arg.identifier().as_string();
        const size_t map_size = python_list::get_list_type_map_size(list_id);
        if (map_size > 0 && map_size <= 32)
        {
          struct sortable_elem
          {
            BigInt key;
            size_t pos;
          };

          std::vector<sortable_elem> elems;
          elems.reserve(map_size);
          bool all_constant_ints = true;

          for (size_t i = 0; i < map_size; ++i)
          {
            const std::string elem_id =
              python_list::get_list_element_id(list_id, i);
            if (elem_id.empty())
            {
              all_constant_ints = false;
              break;
            }

            const symbolt *elem_sym = converter_.find_symbol(elem_id);
            if (
              !elem_sym || !elem_sym->value.is_constant() ||
              !(elem_sym->type.is_signedbv() || elem_sym->type.is_unsignedbv()))
            {
              all_constant_ints = false;
              break;
            }

            BigInt key = binary2integer(elem_sym->value.value().c_str(), true);
            elems.push_back({key, i});
          }

          if (all_constant_ints)
          {
            std::stable_sort(
              elems.begin(),
              elems.end(),
              [](const sortable_elem &a, const sortable_elem &b) {
                if (a.key == b.key)
                  return a.pos < b.pos;
                return a.key < b.key;
              });

            if (fast_path_reverse)
              std::reverse(elems.begin(), elems.end());

            nlohmann::json sorted_list;
            sorted_list["_type"] = "List";
            sorted_list["elts"] = nlohmann::json::array();
            converter_.copy_location_fields_from_decl(call_, sorted_list);
            for (const auto &elem : elems)
            {
              nlohmann::json cst;
              cst["_type"] = "Constant";
              cst["value"] = elem.key.to_int64();
              cst["kind"] = nullptr;
              converter_.copy_location_fields_from_decl(call_, cst);
              sorted_list["elts"].push_back(cst);
            }

            python_list sorted_list_expr(converter_, sorted_list);
            return sorted_list_expr.get();
          }
        }
      }
    }
  }

  // Skip builtin dispatch if the user imported a function with the same name
  // e.g. "from other import sum" defines a user sum that shadows the builtin
  bool is_user_imported =
    converter_.find_imported_symbol(function_id_.to_string()) != nullptr;

  const bool has_user_round =
    !find_function(converter_.ast()["body"], func_name).empty();
  if (
    func_name == "round" && call_.contains("func") &&
    call_["func"].value("_type", "") == "Name" && !has_user_round &&
    !is_user_imported)
  {
    if (call_["args"].empty())
      return converter_.get_exception_handler().gen_exception_raise(
        "TypeError", "round() missing required argument");
    auto arg = call_["args"][0];
    return handle_round(arg);
  }

  // sum() accepts an optional start argument (sum(iterable, start)); accept
  // both 1- and 2-arg forms so the typed dispatch picks sum / sum_float
  // consistently. The other builtins below remain 1-arg only.
  const size_t n_args = call_["args"].size();
  const bool is_sorted_min_max =
    func_name == "min" || func_name == "max" || func_name == "sorted";
  if (
    !is_user_imported && ((is_sorted_min_max && n_args == 1) ||
                          (func_name == "sum" && (n_args == 1 || n_args == 2))))
  {
    exprt list_arg = converter_.get_expr(call_["args"][0]);
    typet elem_type;
    if (list_arg.is_symbol())
    {
      const std::string &list_id = list_arg.identifier().as_string();
      // Check that all elements have the same type and get the common type
      // Returns double_type() for mixed int/float lists (Python semantics)
      elem_type = python_list::check_homogeneous_list_types(list_id, func_name);

      // Mixed int/float list: inline the comparison to avoid type confusion
      // when passing the list to max_float/min_float model functions.
      if (
        elem_type.is_floatbv() && (func_name == "min" || func_name == "max") &&
        python_list::has_mixed_numeric_types(list_id))
      {
        irep_idt comparison_op =
          (func_name == "max") ? exprt::i_gt : exprt::i_lt;
        python_list list_helper(converter_, call_["args"][0]);
        return list_helper.build_min_max_for_mixed_numeric(
          list_arg, list_id, func_name, comparison_op);
      }
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
    obj_symbol = converter_.find_symbol(obj_symbol_id.to_string());
  }

  // Indirect call through variable holding a function pointer, e.g.:
  // times3 = make_multiplier(3); times3(4)
  if (call_["func"]["_type"] == "Name")
  {
    symbol_id var_sid = converter_.create_symbol_id();
    var_sid.set_object(func_name);
    symbolt *var_symbol = symbol_table.find_symbol(var_sid.to_string());
    if (var_symbol && !var_symbol->type.is_code())
    {
      side_effect_expr_function_callt call;
      call.location() = converter_.get_location_from_decl(call_);
      exprt func_expr = symbol_expr(*var_symbol);
      if (
        !var_symbol->type.is_pointer() || !var_symbol->type.subtype().is_code())
        func_expr = typecast_exprt(func_expr, gen_pointer_type(code_typet()));
      call.function() = func_expr;

      bool resolved = false;
      if (
        var_symbol->value.is_address_of() &&
        !var_symbol->value.operands().empty() &&
        var_symbol->value.op0().is_symbol())
      {
        const symbolt *target_symbol =
          symbol_table.find_symbol(var_symbol->value.op0().identifier());
        if (target_symbol && target_symbol->type.is_code())
        {
          call.type() = to_code_type(target_symbol->type).return_type();
          resolved = true;
        }
      }
      if (
        !resolved && var_symbol->type.is_pointer() &&
        var_symbol->type.subtype().is_code())
      {
        call.type() = to_code_type(var_symbol->type.subtype()).return_type();
        resolved = true;
      }
      if (!resolved)
        call.type() = any_type();

      for (const auto &arg_node : call_["args"])
      {
        exprt arg = converter_.get_expr(arg_node);
        if (arg.type().is_code() && arg.is_symbol())
          arg = address_of_exprt(arg);
        call.arguments().push_back(arg);
      }
      return call;
    }
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

  // Find function symbol (O1: use per-call cache to avoid redundant lookups)
  const symbolt *func_symbol = cached_find_symbol(func_symbol_id);

  if (func_symbol == nullptr)
  {
    // Dataclass synthesized constructors may call Class.__post_init__(...) before
    // the class method symbol is fully registered. Preserve class scope and emit
    // a forward reference call instead of falling back to global scope.
    if (
      function_type_ == FunctionType::ClassMethod &&
      function_id_.get_function() == "__post_init__" &&
      !function_id_.get_class().empty())
    {
      locationt location = converter_.get_location_from_decl(call_);
      code_function_callt call;
      call.location() = location;
      call.function() = symbol_exprt(func_symbol_id, code_typet());
      call.type() = empty_typet();

      for (const auto &arg_node : call_["args"])
      {
        exprt arg = converter_.get_expr(arg_node);
        if (arg.type().is_array())
          call.arguments().push_back(address_of_exprt(arg));
        else
          call.arguments().push_back(arg);
      }

      return call;
    }

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

          // If no classes found, use the inferred class name as a best-effort
          // fallback (it may come from weak type inference in dynamic code).
          bool inferred_classes_from_fallback = false;
          if (possible_classes.empty())
          {
            possible_classes.push_back(class_name);
            inferred_classes_from_fallback = true;
          }

          // When there are multiple possible classes (polymorphic object),
          // the method must exist in ALL of them; otherwise it is an
          // AttributeError (the object could be any of those types at
          // runtime and at least one path would fail).
          // For a single class, it suffices that the method exists.
          bool method_exists = false;
          if (possible_classes.size() > 1)
          {
            bool all_have_method = true;
            for (const auto &check_class : possible_classes)
            {
              if (check_class.empty())
                continue;
              if (!method_exists_in_class_hierarchy(check_class, method_name))
              {
                all_have_method = false;
                break;
              }
            }
            method_exists = all_have_method;
          }
          else
          {
            for (const auto &check_class : possible_classes)
            {
              if (check_class.empty())
                continue;
              if (method_exists_in_class_hierarchy(check_class, method_name))
              {
                method_exists = true;
                break;
              }
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
            // In dynamic/untyped flows we may only have fallback class guesses.
            // Do not inject a hard failure from uncertain inference.
            if (inferred_classes_from_fallback)
            {
              locationt location = converter_.get_location_from_decl(call_);
              exprt zero_fallback = gen_zero(any_type());
              zero_fallback.location() = location;
              zero_fallback.location().user_provided(true);
              return zero_fallback;
            }

            // Generate AttributeError for concrete class information.
            return generate_attribute_error(method_name, possible_classes);
          }

          // Method exists or we're in a constructor - create forward reference
          locationt location = converter_.get_location_from_decl(call_);
          code_function_callt call;
          call.location() = location;
          call.function() = symbol_exprt(func_symbol_id, code_typet());
          call.type() = empty_typet();

          if (obj_symbol)
          {
            exprt receiver = symbol_expr(*obj_symbol);
            call.arguments().push_back(
              receiver.type().is_pointer() ? receiver
                                           : gen_address_of(receiver));
          }

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
              else if (arg.is_constant())
              {
                // Constant array (e.g., folded string concat) must be materialized before address_of_exprt.
                symbolt &tmp = converter_.create_tmp_symbol(
                  call_, "$const_str_arg$", arg.type(), arg);
                code_declt tmp_decl(symbol_expr(tmp));
                tmp_decl.location() = location;
                converter_.current_block->copy_to_operands(tmp_decl);
                arg = symbol_expr(tmp);
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
              else if (arg.is_constant())
              {
                // Constant array (e.g., folded string concat) must be materialized before address_of_exprt.
                symbolt &tmp = converter_.create_tmp_symbol(
                  call_, "$const_str_arg$", arg.type(), arg);
                code_declt tmp_decl(symbol_expr(tmp));
                tmp_decl.location() = location;
                converter_.current_block->copy_to_operands(tmp_decl);
                arg = symbol_expr(tmp);
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

  auto bind_instance_receiver = [&](exprt receiver) -> exprt {
    return receiver.type().is_pointer() ? receiver : gen_address_of(receiver);
  };

  auto bind_instance_receiver_symbol =
    [&](const symbolt &receiver_symbol) -> exprt {
    exprt receiver = symbol_expr(receiver_symbol);
    return bind_instance_receiver(receiver);
  };

  // Determine parameter offset for Optional wrapping logic
  size_t param_offset = 0;

  // Add self as first parameter
  if (function_type_ == FunctionType::Constructor)
  {
    // Keep the constructor result as the requested class type, even when
    // __init__ is resolved in a base class.
    const std::string requested_class = function_id_.get_class();
    if (!requested_class.empty())
      call.type() = type_handler_.get_typet(requested_class);
    else
      call.type() = type_handler_.get_typet(func_symbol->name.as_string());

    // Detect super().__init__() pattern: call parent ctor on current self,
    // not on a newly allocated object.
    bool is_super_init = call_["func"]["_type"] == "Attribute" &&
                         call_["func"]["value"]["_type"] == "Call" &&
                         call_["func"]["value"].contains("func") &&
                         call_["func"]["value"]["func"].contains("id") &&
                         call_["func"]["value"]["func"]["id"] == "super";

    if (is_super_init)
    {
      if (obj_symbol)
        call.arguments().push_back(bind_instance_receiver_symbol(*obj_symbol));
      param_offset = 1;
    }
    // Self is the LHS
    else if (converter_.current_lhs)
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
      call.arguments().push_back(bind_instance_receiver_symbol(*obj_symbol));
    }
    else
    {
      // Nested attribute or temporary instance: build expression dynamically
      if (
        call_["func"]["_type"] == "Attribute" &&
        call_["func"].contains("value"))
      {
        const auto &func_value = call_["func"]["value"];
        if (
          func_value["_type"] == "Call" && func_value.contains("func") &&
          func_value["func"]["_type"] == "Name")
        {
          // A().f(...): create a temporary A instance and use it as self.
          const std::string &class_name =
            func_value["func"]["id"].get<std::string>();
          typet class_type = type_handler_.get_typet(class_name);

          symbolt &tmp = converter_.create_tmp_symbol(
            func_value, "$inst$", class_type, exprt());
          converter_.symbol_table().add(tmp);
          code_declt tmp_decl(symbol_expr(tmp));
          tmp_decl.location() = location;
          converter_.current_block->copy_to_operands(tmp_decl);

          // Call the constructor if it is defined, using tmp as self.
          exprt *saved_lhs = converter_.current_lhs;
          exprt tmp_expr = symbol_expr(tmp);
          converter_.current_lhs = &tmp_expr;
          exprt ctor_result = converter_.get_expr(func_value);
          converter_.current_lhs = saved_lhs;

          call.arguments().push_back(bind_instance_receiver(symbol_expr(tmp)));
        }
        else if (func_value["_type"] == "Call")
        {
          // Chained method call (e.g., B().g().f()): the receiver is the return
          // value of an inner method call. Create a temp to hold it and use
          // &temp as self so that self is addressable in the GOTO IR.
          std::string receiver_type =
            type_handler_.get_operand_type(func_value);
          if (!receiver_type.empty())
          {
            typet class_type = type_handler_.get_typet(receiver_type);
            symbolt &tmp = converter_.create_tmp_symbol(
              func_value, "$inst$", class_type, exprt());
            converter_.symbol_table().add(tmp);
            code_declt tmp_decl(symbol_expr(tmp));
            tmp_decl.location() = location;
            converter_.current_block->copy_to_operands(tmp_decl);

            // Process the inner call; set its LHS to tmp so the return value
            // is stored there (emits: FUNCTION_CALL: tmp = inner_call(...)).
            exprt inner_call = converter_.get_expr(func_value);
            if (
              inner_call.is_code() && inner_call.statement() == "function_call")
            {
              inner_call.op0() = symbol_expr(tmp);
              inner_call.location() = location;
              converter_.add_instruction(inner_call);
            }
            call.arguments().push_back(
              bind_instance_receiver(symbol_expr(tmp)));
          }
          else
          {
            exprt obj_expr = converter_.get_expr(func_value);
            call.arguments().push_back(bind_instance_receiver(obj_expr));
          }
        }
        else
        {
          // Member/variable receiver (e.g., self.builder.build()): use the
          // actual receiver expression instead of a nondet temporary.
          exprt obj_expr = converter_.get_expr(func_value);
          call.arguments().push_back(bind_instance_receiver(obj_expr));
        }
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

    // A function name passed as an argument decays to a function pointer,
    // mirroring C's implicit function-to-pointer conversion.
    if (arg.type().is_code() && arg.is_symbol())
      arg = address_of_exprt(arg);

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

      // Handle struct argument passed to a union-typed parameter (e.g. str | T).
      // Union parameters are stored as pointer(char[0]). When the actual argument
      // is a struct (class instance), take its address and cast to the pointer type
      // so that the attribute access handler can safely cast back and dereference.
      // Follow symbol types because class symbols use symbol_typet, not struct_typet.
      // NOTE: python_converter.cpp has a complementary post-processing pass that
      // handles the general pointer-to-struct coercion case. This earlier pass is
      // specific to the char[0]* union representation and materialises non-symbol
      // struct temporaries before taking their address.
      typet arg_followed_type = converter_.ns.follow(arg.type());
      if (
        param_type.is_pointer() && param_type.subtype().is_array() &&
        param_type.subtype().subtype() == char_type() &&
        arg_followed_type.is_struct())
      {
        if (!arg.is_symbol())
        {
          // Materialize the struct in a temp variable first
          symbolt &tmp = converter_.create_tmp_symbol(
            call_, "$union_arg$", arg.type(), gen_zero(arg.type()));
          code_declt tmp_decl(symbol_expr(tmp));
          tmp_decl.location() = location;
          converter_.current_block->copy_to_operands(tmp_decl);
          code_assignt tmp_assign(symbol_expr(tmp), arg);
          tmp_assign.location() = location;
          converter_.current_block->copy_to_operands(tmp_assign);
          arg = symbol_expr(tmp);
        }
        arg = typecast_exprt(address_of_exprt(arg), param_type);
      }

      // General object-reference coercion:
      // when a pointer parameter receives a struct object argument, pass the
      // object's address (materializing temporaries when required).
      if (
        function_type_ == FunctionType::Constructor &&
        param_type.is_pointer() && arg_followed_type.is_struct() &&
        !arg.is_address_of())
      {
        if (!arg.is_symbol())
        {
          symbolt &tmp = converter_.create_tmp_symbol(
            call_, "$ptr_arg$", arg.type(), gen_zero(arg.type()));
          code_declt tmp_decl(symbol_expr(tmp));
          tmp_decl.location() = location;
          converter_.current_block->copy_to_operands(tmp_decl);
          code_assignt tmp_assign(symbol_expr(tmp), arg);
          tmp_assign.location() = location;
          converter_.current_block->copy_to_operands(tmp_assign);
          arg = symbol_expr(tmp);
        }

        arg = address_of_exprt(arg);
        if (!base_type_eq(arg.type(), param_type, converter_.ns))
          arg = typecast_exprt(arg, param_type);
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
      (function_id_.get_function() == "__ESBMC_get_object_size" ||
       function_id_.get_function() == "strlen") &&
      (arg.type() == type_handler_.get_list_type() ||
       (arg.type().is_pointer() &&
        arg.type().subtype() == type_handler_.get_list_type() &&
        arg.type().is_symbol())))
    {
      symbolt *list_symbol = nullptr;

      if (arg.is_symbol())
      {
        list_symbol = converter_.find_symbol(arg.identifier().as_string());
      }
      else
      {
        const typet list_type = type_handler_.get_list_type();
        symbolt &tmp_list = converter_.create_tmp_symbol(
          call_, "$obj_size_list_arg$", list_type, exprt());

        code_declt tmp_decl(symbol_expr(tmp_list));
        tmp_decl.location() = location;
        converter_.current_block->copy_to_operands(tmp_decl);

        code_assignt tmp_assign(symbol_expr(tmp_list), arg);
        tmp_assign.location() = location;
        converter_.current_block->copy_to_operands(tmp_assign);

        list_symbol = &tmp_list;
      }

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
      else if (arg.is_constant())
      {
        // Constant array (e.g., folded string concat) must be materialized before address_of_exprt.
        symbolt &tmp = converter_.create_tmp_symbol(
          call_, "$const_str_arg$", arg.type(), arg);
        code_declt tmp_decl(symbol_expr(tmp));
        tmp_decl.location() = location;
        converter_.current_block->copy_to_operands(tmp_decl);
        arg = symbol_expr(tmp);
      }
      call.arguments().push_back(address_of_exprt(arg));
    }
    else
      call.arguments().push_back(arg);

    arg_index++;
  }

  // Forward keyword arguments to their parameter slots so the callee
  // receives the supplied value. The validation loop below only fills in
  // default values for *missing* params, so kwargs would otherwise be
  // marked "provided" yet never actually passed.
  if (call_.contains("keywords") && call_["keywords"].is_array())
  {
    for (const auto &kw : call_["keywords"])
    {
      if (!kw.contains("arg") || kw["arg"].is_null())
        continue; // skip **kwargs unpacking
      const std::string kw_name = kw["arg"].get<std::string>();
      for (size_t i = 0; i < params.size(); ++i)
      {
        if (params[i].get_base_name().as_string() == kw_name)
        {
          exprt kw_val = converter_.get_expr(kw["value"]);
          if (call.arguments().size() <= i)
            call.arguments().resize(i + 1);
          call.arguments()[i] = kw_val;
          break;
        }
      }
    }
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
  if (function_type_ == FunctionType::Constructor && !converter_.current_lhs)
  {
    size_t num_provided_args = call_["args"].size();

    // Only add self if arguments size matches user args (no self added yet)
    if (call.arguments().size() == num_provided_args)
    {
      // Create temporary object as self parameter
      const std::string requested_class = function_id_.get_class();
      typet class_type =
        requested_class.empty()
          ? type_handler_.get_typet(func_symbol->name.as_string())
          : type_handler_.get_typet(requested_class);
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

      // Emit the constructor call directly as a FUNCTION_CALL instruction so
      // ESBMC inlines it (codet("expression") would produce OTHER and be
      // skipped).  Return the initialised temp_self symbol so callers such as
      // list literals receive the properly constructed object.
      call.location() = location;
      converter_.add_instruction(call);
      return symbol_expr(temp_self);
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

  // Build cache key from symbol id; fallback to function + symbol name.
  std::string cache_key = obj_symbol->id.as_string();
  if (cache_key.empty())
    cache_key =
      converter_.current_function_name() + "::" + obj_symbol->name.as_string();

  // Check cache first.
  if (
    auto cached =
      converter_.get_function_call_cache().get_possible_class_types(cache_key))
    return *cached;

  try
  {
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
      converter_.get_function_call_cache().set_possible_class_types(
        cache_key, possible_classes);
      return possible_classes;
    }

    // Type is a primitive (e.g., floatbv) - trace through AST to find actual types
    std::string var_name = obj_symbol->name.as_string();
    nlohmann::json var_decl = json_utils::find_var_decl(
      var_name, converter_.current_function_name(), converter_.ast());

    if (
      var_decl.empty() || !var_decl.is_object() || !var_decl.contains("value"))
    {
      converter_.get_function_call_cache().set_possible_class_types(
        cache_key, possible_classes);
      return possible_classes;
    }

    const auto &value = var_decl["value"];

    // Validate JSON shape before accessing nested fields.
    if (
      !value.is_object() || !value.contains("_type") ||
      !value["_type"].is_string())
    {
      converter_.get_function_call_cache().set_possible_class_types(
        cache_key, possible_classes);
      return possible_classes;
    }

    // Check if assigned from a function call
    if (value["_type"] != "Call")
    {
      converter_.get_function_call_cache().set_possible_class_types(
        cache_key, possible_classes);
      return possible_classes;
    }

    if (
      !value.contains("func") || !value["func"].is_object() ||
      !value["func"].contains("_type") || !value["func"]["_type"].is_string() ||
      value["func"]["_type"] != "Name" || !value["func"].contains("id") ||
      !value["func"]["id"].is_string())
    {
      converter_.get_function_call_cache().set_possible_class_types(
        cache_key, possible_classes);
      return possible_classes;
    }

    std::string func_name = value["func"]["id"].get<std::string>();

    // Look up the function definition
    const auto &func_node =
      json_utils::find_function(converter_.ast()["body"], func_name);

    if (
      func_node.empty() || !func_node.is_object() ||
      !func_node.contains("returns") || func_node["returns"].is_null())
    {
      converter_.get_function_call_cache().set_possible_class_types(
        cache_key, possible_classes);
      return possible_classes;
    }

    const auto &returns = func_node["returns"];
    if (
      !returns.is_object() || !returns.contains("_type") ||
      !returns["_type"].is_string() || returns["_type"] != "Name" ||
      !returns.contains("id") || !returns["id"].is_string())
    {
      converter_.get_function_call_cache().set_possible_class_types(
        cache_key, possible_classes);
      return possible_classes;
    }

    std::string return_type = returns["id"].get<std::string>();

    // If return type is 'Any', analyze the function body to find actual return classes
    if (
      return_type == "Any" && func_node.contains("body") &&
      func_node["body"].is_array())
    {
      std::function<void(const nlohmann::json &)> find_returns;
      find_returns = [&](const nlohmann::json &node) {
        if (
          !node.is_object() || !node.contains("_type") ||
          !node["_type"].is_string())
          return;

        std::string node_type = node["_type"].get<std::string>();

        if (
          node_type == "Return" && node.contains("value") &&
          node["value"].is_object())
        {
          const auto &ret_val = node["value"];
          if (
            ret_val.contains("_type") && ret_val["_type"].is_string() &&
            ret_val["_type"] == "Call" && ret_val.contains("func") &&
            ret_val["func"].is_object() && ret_val["func"].contains("_type") &&
            ret_val["func"]["_type"].is_string() &&
            ret_val["func"]["_type"] == "Name" &&
            ret_val["func"].contains("id") && ret_val["func"]["id"].is_string())
          {
            std::string class_name = ret_val["func"]["id"].get<std::string>();
            if (json_utils::is_class(class_name, converter_.ast()))
              possible_classes.push_back(class_name);
          }
        }
        else if (node_type == "If")
        {
          // Check both branches
          if (node.contains("body") && node["body"].is_array())
            for (const auto &stmt : node["body"])
              find_returns(stmt);
          if (node.contains("orelse") && node["orelse"].is_array())
            for (const auto &stmt : node["orelse"])
              find_returns(stmt);
        }
      };

      for (const auto &stmt : func_node["body"])
        find_returns(stmt);
    }
  }
  catch (...)
  {
    // Malformed AST — return whatever we gathered so far.
  }

  // Remove empty class names before caching.
  possible_classes.erase(
    std::remove_if(
      possible_classes.begin(),
      possible_classes.end(),
      [](const std::string &s) { return s.empty(); }),
    possible_classes.end());

  // Deduplicate while preserving order of appearance in the AST.
  {
    std::vector<std::string> unique;
    std::unordered_set<std::string> seen;
    for (const auto &cls : possible_classes)
    {
      if (seen.insert(cls).second)
        unique.push_back(cls);
    }
    possible_classes = std::move(unique);
  }

  converter_.get_function_call_cache().set_possible_class_types(
    cache_key, possible_classes);
  return possible_classes;
}

bool function_call_expr::method_exists_in_class_hierarchy(
  const std::string &class_name,
  const std::string &method_name) const
{
  // Reject empty inputs early.
  if (class_name.empty() || method_name.empty())
    return false;

  std::string cache_key = class_name + "::" + method_name;

  // Cache lookup first.
  auto cached =
    converter_.get_function_call_cache().get_method_exists(cache_key);
  if (cached.has_value())
    return cached.value();

  // Provisional negative cache write to break recursive cycles (A->B->A).
  converter_.get_function_call_cache().set_method_exists(cache_key, false);

  const auto &ast = converter_.ast();
  if (!ast.is_object() || !ast.contains("body") || !ast["body"].is_array())
    return false;

  const auto &class_node = json_utils::find_class(ast["body"], class_name);

  if (class_node.empty() || !class_node.is_object())
  {
    // Already cached as false.
    return false;
  }

  // Check only top-level class methods (FunctionDef / AsyncFunctionDef).
  if (class_node.contains("body") && class_node["body"].is_array())
  {
    for (const auto &member : class_node["body"])
    {
      if (
        !member.is_object() || !member.contains("_type") ||
        !member["_type"].is_string())
        continue;

      const std::string &member_type = member["_type"].get<std::string>();
      if (
        (member_type == "FunctionDef" || member_type == "AsyncFunctionDef") &&
        member.contains("name") && member["name"].is_string() &&
        member["name"].get<std::string>() == method_name)
      {
        // Method found — upgrade cache to true.
        converter_.get_function_call_cache().set_method_exists(cache_key, true);
        return true;
      }
    }
  }

  // Traverse bases — only if base node is valid and base id is non-empty.
  if (class_node.contains("bases") && class_node["bases"].is_array())
  {
    for (const auto &base : class_node["bases"])
    {
      if (!base.is_object() || !base.contains("id") || !base["id"].is_string())
        continue;

      std::string base_name = base["id"].get<std::string>();
      if (base_name.empty())
        continue;

      if (method_exists_in_class_hierarchy(base_name, method_name))
      {
        // Found in ancestor — upgrade cache to true.
        converter_.get_function_call_cache().set_method_exists(cache_key, true);
        return true;
      }
    }
  }

  return false;
}

exprt function_call_expr::generate_attribute_error(
  const std::string &method_name,
  const std::vector<std::string> &possible_classes,
  const typet &expected_type) const
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

  converter_.add_instruction(assert_code);

  // Compute fallback type: use expected_type if valid, otherwise Any
  typet fallback_type = expected_type;
  if (
    fallback_type.is_nil() || fallback_type == empty_typet() ||
    fallback_type == typet())
    fallback_type = any_type();

  exprt nondet_fallback("sideeffect", fallback_type);
  nondet_fallback.statement("nondet");
  nondet_fallback.location() = location;
  nondet_fallback.location().user_provided(true);
  nondet_fallback.location().comment(error_msg.str());

  return nondet_fallback;
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
