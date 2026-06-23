#include <python-frontend/function_call/expr.h>
#include <python-frontend/cmath_lowering_policy.h>
#include <python-frontend/complex_handler.h>
#include <python-frontend/complex_handler_utils.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/math_guard_utils.h>
#include <python-frontend/python_exception_handler.h>
#include <python-frontend/python_list.h>
#include <python-frontend/python_set.h>
#include <python-frontend/string/string_builder.h>
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
#include <util/c_sizeof.h>
#include <util/string_constant.h>
#include <irep2/irep2_utils.h>
#include <util/migrate.h>

#include <algorithm>
#include <cctype>
#include <optional>
#include <regex>
#include <stdexcept>
#include <unordered_set>

using namespace json_utils;
namespace
{
// V.3: IREP2 expression-construction helpers (exact round-trip; behaviour-
// preserving). Back-migrated for the legacy adjust/goto-convert seam.
//
// A dynamically-sized array type (non-constant size) does not survive the
// migrate_type round-trip (get_width throws downstream), so the helpers fall
// back to the legacy constructor when the relevant type contains one.
bool contains_dyn_array(const typet &t)
{
  if (t.is_array())
  {
    const array_typet &at = to_array_type(t);
    if (at.size().is_nil() || !at.size().is_constant())
      return true;
    return contains_dyn_array(at.subtype());
  }
  if (t.is_pointer())
    return contains_dyn_array(t.subtype());
  return false;
}

exprt build_symbol(const symbolt &sym)
{
  if (contains_dyn_array(sym.get_type()))
    return symbol_expr(sym);
  return migrate_expr_back(symbol_expr2tc(sym));
}

exprt build_typecast(const exprt &from, const typet &t)
{
  if (contains_dyn_array(t) || contains_dyn_array(from.type()))
    return typecast_exprt(from, t);
  expr2tc from2;
  migrate_expr(from, from2);
  exprt result = migrate_expr_back(typecast2tc(migrate_type(t), from2));
  // migrate_type does not round-trip type attributes such as #cpp_type;
  // restore the exact target type so legacy typecast_exprt(from, t) is
  // reproduced faithfully.
  result.type() = t;
  return result;
}

exprt build_address_of(const exprt &obj)
{
  if (contains_dyn_array(obj.type()))
    return address_of_exprt(obj);
  expr2tc obj2;
  migrate_expr(obj, obj2);
  return migrate_expr_back(address_of2tc(obj2->type, obj2));
}

// Struct member access base.field : t (V.3). `base` must be a struct/union/
// complex value (member2t's source precondition); the callers here pass a
// tuple or complex struct whose component is named `name`.
exprt build_member(const exprt &base, const irep_idt &name, const typet &t)
{
  if (contains_dyn_array(t) || contains_dyn_array(base.type()))
    return member_exprt(base, name, t);
  expr2tc base2;
  migrate_expr(base, base2);
  exprt result = migrate_expr_back(member2tc(migrate_type(t), base2, name));
  // migrate_type does not round-trip #cpp_type; restore the exact member type
  // so legacy member_exprt(base, name, t) is reproduced faithfully.
  result.type() = t;
  return result;
}
} // namespace

namespace
{
// Constants for UTF-8 encoding

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

} // namespace

bool is_cpp_throw_expr(const exprt &e)
{
  return e.statement() == "cpp-throw";
}

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

void function_call_expr::get_function_type()
{
  const auto &func_node = call_["func"];

  // An explicit `Base.__init__(self, ...)` call invokes the named base class's
  // constructor with self passed explicitly; it is not an object construction.
  // Let it fall through to the ClassMethod classification (is_class(caller)
  // below) so no fresh self object is allocated -- the builder resolves it to
  // the class's renamed constructor (@C@Base@F@Base) and the explicit self is
  // the receiver. Without this it would be classified Constructor and the
  // constructor would write to a throwaway $ctor_self$ temp.
  const bool is_explicit_class_init =
    func_node.contains("_type") && func_node["_type"] == "Attribute" &&
    func_node.contains("attr") && func_node["attr"] == "__init__" &&
    func_node.contains("value") && func_node["value"].is_object() &&
    func_node["value"].contains("_type") &&
    func_node["value"]["_type"] == "Name" &&
    func_node["value"].contains("id") &&
    json_utils::is_class(
      func_node["value"]["id"].get<std::string>(), converter_.ast());

  if (!is_explicit_class_init && type_handler_.is_constructor_call(call_))
  {
    function_type_ = FunctionType::Constructor;
    return;
  }

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

  // tuple(iterable): model the result as a shallow copy of the underlying
  // list (CPython copy semantics: later mutations of the source list must
  // not show through the tuple; element references are shared, not deep-
  // copied). ==, len(), subscript and iteration then route through the
  // existing list machinery. The generic constructor tail below would
  // instead relabel the expression with get_typet("tuple") — an empty type —
  // and every comparison over the result silently lowers to the nondet-bool
  // fallback in get_binary_operator_expr (#4807).
  if (func_name == "tuple")
  {
    if (call_["args"].size() > 1)
      throw std::runtime_error("TypeError: tuple expected at most 1 argument");
    exprt expr = converter_.get_expr(arg);
    const typet &et = expr.type();
    if (converter_.get_tuple_handler().is_tuple_type(et))
      return expr; // returned unchanged — CPython tuple(t) also returns t
    const namespacet ns(converter_.symbol_table());
    const typet list_type = type_handler_.get_list_type(); // PyListObject *
    if (
      et == list_type ||
      (et.is_pointer() && list_type.is_pointer() &&
       ns.follow(et.subtype()) == ns.follow(list_type.subtype())))
    {
      // Covers list literals, variables, and list-returning calls.
      python_list list_handler(converter_, call_);
      return list_handler.build_shallow_copy_call(expr, call_);
    }
    throw std::runtime_error(
      "tuple() is only supported over list and tuple arguments");
  }

  // bytes(...) constructor. The generic constructor path below relabels the
  // argument expression's type as the bytes array type without converting the
  // value; for a list/int argument that yields a list-pointer (or scalar) value
  // tagged as an array, which trips base_type_eq in value_set (a crash). Build
  // a real byte array here instead, matching the bytes-literal representation.
  if (func_name == "bytes")
  {
    // bytes([i0, i1, ...]) — a list of constant ints in range(0, 256).
    if (
      arg.is_object() && arg.value("_type", "") == "List" &&
      arg.contains("elts"))
    {
      std::vector<uint8_t> bytes;
      for (const auto &e : arg["elts"])
      {
        if (
          !e.is_object() || e.value("_type", "") != "Constant" ||
          !e.contains("value") || !e["value"].is_number_integer())
          throw std::runtime_error(
            "bytes(): only a list of constant integers is supported");
        const long long v = e["value"].get<long long>();
        if (v < 0 || v > 255)
          throw std::runtime_error(
            "ValueError: bytes must be in range(0, 256)");
        bytes.push_back(static_cast<uint8_t>(v));
      }
      // An empty byte array is modelled as a size-0 (variable-length) array,
      // which len()/iteration then route through strlen; reuse the no-argument
      // bytes() representation, which the size-0 path handles correctly.
      if (bytes.empty())
        return exprt("constant", type_handler_.get_typet("bytes", 0));
      return converter_.get_string_builder().build_raw_byte_array(bytes);
    }

    // bytes(n) — a constant non-negative count of zero bytes.
    if (
      arg.is_object() && arg.value("_type", "") == "Constant" &&
      arg.contains("value") && arg["value"].is_number_integer())
    {
      const long long n = arg["value"].get<long long>();
      if (n < 0)
        throw std::runtime_error("ValueError: negative count");
      if (n == 0)
        return exprt("constant", type_handler_.get_typet("bytes", 0));
      return converter_.get_string_builder().build_raw_byte_array(
        std::vector<uint8_t>(static_cast<size_t>(n), 0));
    }
  }

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
      // The compile-time fast path below decodes the symbol's stored value as
      // a string; it is only valid for constant *string* symbols. For numeric
      // symbols (int/float/bool) extract_string_from_symbol misreads the value
      // — an int 65 decodes to the character 'A' (rejected as non-digit) and a
      // float yields no string at all — so int(x) wrongly folds to 0. Route
      // numeric symbols through the general numeric conversion instead, which
      // truncates floats toward zero and treats ints as identity. (GitHub #4770)
      if (
        sym && sym->get_value().is_constant() &&
        type_utils::is_string_type(sym->get_type()))
      {
        if (base_expr.is_nil())
        {
          return handle_str_symbol_to_int(sym);
        }
        else
        {
          // Convert symbol to expression and use with base
          exprt value_expr = build_symbol(*sym);
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
    // Only take the constant-string fast path when the variable is genuinely
    // string-typed here. A numeric variable conditionally reassigned a string
    // (e.g. `if isinstance(x, str): x = x.replace(...)`) leaves a stale string
    // value on its symbol even on paths where it stays numeric; keying off the
    // value alone would mis-route float(x) into string parsing and fold it to
    // 0.0 (#5161). Gate on the declared type so such cases fall through to the
    // numeric typecast in the else branch.
    if (
      sym && sym->get_value().is_constant() &&
      type_utils::is_string_type(sym->get_type()) &&
      type_utils::is_string_type(sym->get_value().type()))
      return handle_str_symbol_to_float(sym);
    else
    {
      // Try to get the expression type directly, even if symbol lookup failed
      exprt expr = converter_.get_expr(arg);
      if (type_utils::is_string_type(expr.type()))
      {
        // Runtime string -> float. float("10") must succeed, but float() of an
        // arbitrary string may raise ValueError. Gate the conversion on a
        // runtime validity check so a concrete valid literal folds away while a
        // genuinely non-float string still raises a reachable ValueError (the
        // same exception path used by the string-literal case above).
        auto &sh = converter_.get_string_handler();
        auto loc = converter_.get_location_from_decl(call_);

        exprt valid = sh.handle_string_is_float(expr, loc);
        exprt raise = converter_.get_exception_handler().gen_exception_raise(
          "ValueError", "could not convert string to float");
        codet throw_code("expression");
        throw_code.operands().push_back(raise);
        code_ifthenelset guard;
        // V.3: build the "not valid" guard condition in IREP2.
        expr2tc valid2;
        migrate_expr(valid, valid2);
        guard.cond() = migrate_expr_back(not2tc(valid2));
        guard.then_case() = throw_code;
        guard.location() = loc;
        converter_.add_instruction(guard);

        return sh.handle_string_to_float(expr, loc);
      }
      // Numeric variable: emit a proper typecast to avoid mislabeled IR
      typet float_t = type_handler_.get_typet("float", 0);
      if (!expr.type().is_floatbv())
        return build_typecast(expr, float_t);
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

  // Handle bin: Handles binary string arguments
  else if (func_name == "bin")
    return handle_bin(arg);

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

      // Numeric / boolean variables: convert_to_string folds constants to a
      // char array literal and dispatches non-constants to the matching
      // __python_*_to_str operational model. Strings flow through unchanged.
      const typet &vt = value_expr.type();
      if (
        vt.is_bool() || type_utils::is_integer_type(vt) || vt.is_floatbv() ||
        type_utils::is_string_type(vt))
        return converter_.get_string_handler().convert_to_string(value_expr);

      // Element of a list whose element type could not be statically resolved
      // (e.g. iterating an empty/untyped list) is typed as the generic list
      // pointer. Treat it as the documented list[int] default so str() lowers
      // through the integer model instead of aborting the whole run; the loop
      // body is dead for the empty list, mirroring the arithmetic coercion in
      // get_binary_operator_expr. Other unresolved pointers (e.g. an
      // unannotated void* parameter) fall through to the sound nondet-string
      // fallback below rather than being guessed as int.
      if (vt == type_handler_.get_list_type())
        return converter_.get_string_handler().convert_to_string(
          build_typecast(value_expr, type_handler_.get_typet("int", 0)));
    }

    // A string literal argument folds to its length below.
    if (arg.contains("value") && arg["value"].is_string())
      arg_size = handle_str(arg);
    else
      // The argument has no statically stringifiable type. This happens for an
      // unannotated parameter, modelled as Any (void*): its dynamic type is
      // unknown, so str() cannot be folded. Fall back to a sound nondet string
      // rather than aborting conversion of the whole program — subsequent ops
      // see arbitrary content, which over-approximates str() of an unknown
      // value without wrongly concluding any specific result.
      return converter_.get_string_handler().build_nondet_string_fallback(
        converter_.get_location_from_decl(call_));
  }

  typet t = type_handler_.get_typet(func_name, arg_size);
  exprt expr = converter_.get_expr(arg);

  // For float(), emit a proper typecast instead of relabeling the type.
  // Simply changing expr.type() on an integer expression creates IR where
  // the type tag says float but the operation is bitvector arithmetic,
  // causing sort mismatches in the SMT encoder.
  if (func_name == "float" && !expr.type().is_floatbv())
    return build_typecast(expr, t);

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

    // Nested list (list-of-list) path: only meaningful when the base is itself
    // a list, since list_type_map keys list-of-list types.  For non-list bases
    // (e.g. dicts whose value is a list) we fall through to the get_expr
    // dispatch below, which can also resolve dict-subscript receivers.
    if (base_sym->get_type() == list_type)
    {
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
        if (sym && sym->get_type() == list_type)
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

    // Dict subscript whose value is a list (e.g. d[k].append(v) where
    // d is a dict[K, list[V]] or a defaultdict(list)).  The dict-subscript
    // expression returns the stored PyListObject pointer; wrap it in a
    // temp symbol so list method handlers can treat it as a named list.
    // List mutations through the temp alias the dict slot because lists in
    // the Python model are reference-typed (PyListObject *).  Mirrors the
    // $attr_list$ / $call_list$ pattern below.
    const exprt subscript_expr = converter_.get_expr(func_value);
    if (subscript_expr.type() == list_type)
    {
      symbolt &tmp = converter_.create_tmp_symbol(
        call_, "$dict_list$", list_type, subscript_expr);
      std::string idx_str = "(expr)";
      if (slice_node.contains("id"))
        idx_str = slice_node["id"].get<std::string>();
      else if (
        slice_node["_type"] == "Constant" &&
        slice_node["value"].is_number_integer())
        idx_str = std::to_string(slice_node["value"].get<size_t>());
      display_name = base_name + "[" + idx_str + "]";
      return &tmp;
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
      if (sym && sym->get_type() == list_type)
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
  if (!sym || sym->get_value().is_nil())
    return;
  // Only emit a declaration for temp symbols produced by
  // get_object_list_symbol() from non-named receivers.  These are identified
  // by the prefixes "$attr_list$", "$call_list$", and "$dict_list$".
  // Regular list symbols have a nil value and are declared elsewhere; this
  // guard prevents re-declaring them.
  const std::string &name = sym->name.as_string();
  if (
    name.find("$attr_list$") == std::string::npos &&
    name.find("$call_list$") == std::string::npos &&
    name.find("$dict_list$") == std::string::npos)
    return;
  code_declt decl(build_symbol(*sym));
  decl.copy_to_operands(sym->get_value());
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
      func_call.type() = to_code_type(sym->get_type()).return_type();
  }
  if (func_call.type().is_nil() || func_call.type().id() == "empty")
    func_call.type() = arg.type();
  return func_call;
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
    code_declt insert_value(build_symbol(insert_value_symbol));
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
  clear_call.function() = build_symbol(*clear_func);
  clear_call.arguments().push_back(build_symbol(*list_symbol));
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

exprt function_call_expr::handle_list_popleft() const
{
  // collections.deque.popleft(): remove and return the front element.
  // deque is modelled as a list, so this is pop(0).
  if (!call_["args"].empty())
    throw std::runtime_error("popleft() takes no arguments");

  std::string list_display_name;
  const symbolt *list_symbol = get_object_list_symbol(list_display_name);
  materialize_list_symbol(list_symbol);
  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_display_name);

  // build_pop_list_call infers the popped element's compile-time type from the
  // back of the type map (it was written for pop()); for a homogeneous deque
  // — the FIFO use case (e.g. breadth_first_search) — front and back share a
  // type, so this is exact. A heterogeneous deque could mis-type the front
  // element; that richer case is left to a future index-aware type map.
  python_list list_helper(converter_, call_);
  return list_helper.build_pop_list_call(
    *list_symbol, from_integer(0, signedbv_typet(64)), call_);
}

exprt function_call_expr::handle_list_appendleft() const
{
  // collections.deque.appendleft(x): prepend x. deque is modelled as a list,
  // so this is insert(0, x).
  const auto &args = call_["args"];
  if (args.size() != 1)
    throw std::runtime_error("appendleft() takes exactly one argument");

  std::string list_display_name;
  const symbolt *list_symbol = get_object_list_symbol(list_display_name);
  materialize_list_symbol(list_symbol);
  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_display_name);

  exprt index_expr = from_integer(0, signedbv_typet(64));
  exprt value_to_insert = converter_.get_expr(args[0]);

  if (value_to_insert.is_constant())
  {
    symbolt &insert_value_symbol = converter_.create_tmp_symbol(
      call_, "insert_value", size_type(), gen_zero(size_type()));
    code_declt insert_value(build_symbol(insert_value_symbol));
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
    // sum(t.element_i == elem ? 1 : 0), built in IREP2 (V.3).
    const type2tc result_type = migrate_type(int_type());
    expr2tc elem2;
    migrate_expr(elem, elem2);
    expr2tc total = gen_zero(result_type);
    for (const auto &comp : components)
    {
      expr2tc member2;
      migrate_expr(
        build_member(receiver, comp.get_name(), comp.type()), member2);
      expr2tc sel = if2tc(
        result_type,
        equality2tc(member2, elem2),
        gen_one(result_type),
        gen_zero(result_type));
      total = add2tc(result_type, total, sel);
    }
    return migrate_expr_back(total);
  }

  // method_name == "index"
  // Return the smallest k for which t.element_k == elem; assert if absent.
  if (components.empty())
    throw std::runtime_error("tuple.index() on empty tuple");

  // Build "any matched" guard so we can assert the element is present (V.3).
  const type2tc result_type = migrate_type(int_type());
  expr2tc elem2;
  migrate_expr(elem, elem2);
  expr2tc any_match = gen_false_expr();
  for (const auto &comp : components)
  {
    expr2tc member2;
    migrate_expr(build_member(receiver, comp.get_name(), comp.type()), member2);
    any_match = or2tc(any_match, equality2tc(member2, elem2));
  }
  code_assertt found_assert(migrate_expr_back(any_match));
  found_assert.location() = converter_.get_location_from_decl(call_);
  found_assert.location().comment("ValueError: tuple.index(x): x not in tuple");
  converter_.add_instruction(found_assert);

  // Build chain right-to-left: result_(n-1) is index n-1, falling back to
  // earlier matches as we walk backwards. Net effect: leftmost match wins.
  size_t n = components.size();
  expr2tc result = from_integer(BigInt(n - 1), result_type);
  for (size_t k = n - 1; k-- > 0;)
  {
    expr2tc member2;
    migrate_expr(
      build_member(receiver, components[k].get_name(), components[k].type()),
      member2);
    result = if2tc(
      result_type,
      equality2tc(member2, elem2),
      from_integer(BigInt(k), result_type),
      result);
  }
  return migrate_expr_back(result);
}

bool function_call_expr::is_dict_method_call() const
{
  if (call_["func"]["_type"] != "Attribute")
    return false;

  const std::string &method_name = function_id_.get_function();

  if (
    !python_dict_handler::is_value_returning_method(method_name) &&
    method_name != "update" && method_name != "clear")
    return false;

  // A receiver that resolves to a non-dict object (e.g. a class instance whose
  // own method shadows the same-named dict method, such as queue.Queue.get())
  // must defer to instance-method dispatch. Real dicts carry the
  // "__python_dict__" struct tag; class instances carry "tag-<Class>". An
  // unresolved or genuinely dict-typed receiver is left to the dict handler, so
  // existing dict dispatch is unchanged. Mirrors the list guard for pop below.
  if (receiver_is_non_dict_object())
    return false;

  // For "pop", which exists on both list and dict, treat as dict.pop() when
  // the receiver does not resolve to a list symbol.
  if (method_name == "pop")
  {
    std::string dummy;
    const symbolt *sym = get_object_list_symbol(dummy);
    const typet list_type = type_handler_.get_list_type();
    return sym == nullptr || sym->get_type() != list_type;
  }

  return true;
}

bool function_call_expr::receiver_is_non_dict_object() const
{
  const auto &recv = call_["func"]["value"];
  if (recv["_type"] != "Name" || !recv.contains("id"))
    return false;

  // Resolve the receiver name in function then module scope. This mirrors
  // lookup_python_symbol but is kept warning-free: is_dict_method_call is a
  // discriminator, so a miss here must stay silent (a genuine dict whose
  // receiver does not resolve through these scopes still defers to the dict
  // handler below — the safe direction).
  const std::string var_name = recv["id"].get<std::string>();
  const std::string filename = function_id_.get_filename();
  const symbolt *sym = converter_.find_symbol(
    "py:" + filename + "@F@" + converter_.current_function_name() + "@" +
    var_name);
  if (!sym)
    sym = converter_.find_symbol("py:" + filename + "@" + var_name);
  if (!sym)
    return false;

  typet t = sym->get_type();
  if (t.is_pointer())
    t = t.subtype();
  if (t.id() == "symbol")
    t = converter_.ns.follow(t);

  // Only a positively-resolved non-dict struct defers to instance dispatch; an
  // unresolved or "__python_dict__"-tagged receiver stays with the dict handler.
  // list/set receivers also resolve to a (non-dict) struct here, but their
  // dict-overlapping methods (pop/copy/update) are claimed by the list/set
  // discriminators earlier in the dispatch table, so they never reach this.
  if (!t.is_struct())
    return false;
  return to_struct_type(t).tag().as_string() != "__python_dict__";
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
    dict_expr = build_symbol(*dict_symbol);
  }
  else
  {
    exprt literal = converter_.get_expr(call_["func"]["value"]);
    symbolt &tmp = converter_.create_tmp_symbol(
      call_, "$dict_lit$", literal.type(), exprt());
    converter_.add_instruction(code_declt(build_symbol(tmp)));
    converter_.add_instruction(code_assignt(build_symbol(tmp), literal));
    dict_expr = build_symbol(tmp);
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

  if (method_name == "clear")
    return converter_.get_dict_handler()->handle_dict_clear(dict_expr, call_);

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

exprt function_call_expr::handle_list_count() const
{
  const auto &args = call_["args"];
  if (args.size() != 1)
    throw std::runtime_error("list.count() takes exactly one argument");

  std::string list_display_name;
  const symbolt *list_symbol = get_object_list_symbol(list_display_name);
  materialize_list_symbol(list_symbol);
  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_display_name);

  exprt value = converter_.get_expr(args[0]);
  python_list list_helper(converter_, call_);
  return list_helper.build_count_list_call(*list_symbol, call_, value);
}

exprt function_call_expr::handle_list_index() const
{
  const auto &args = call_["args"];
  if (args.size() != 1)
    throw std::runtime_error("list.index() takes exactly one argument");

  std::string list_display_name;
  const symbolt *list_symbol = get_object_list_symbol(list_display_name);
  materialize_list_symbol(list_symbol);
  if (!list_symbol)
    throw std::runtime_error("List variable not found: " + list_display_name);

  exprt value = converter_.get_expr(args[0]);
  python_list list_helper(converter_, call_);
  return list_helper.build_index_list_call(*list_symbol, call_, value);
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
  sort_call.function() = build_symbol(*sort_func);
  sort_call.arguments().push_back(build_symbol(*list_symbol));
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
  reverse_call.function() = build_symbol(*reverse_func);
  reverse_call.arguments().push_back(build_symbol(*list_symbol));
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
  reverse_call.function() = build_symbol(*reverse_func);
  reverse_call.arguments().push_back(build_symbol(*list_symbol));
  reverse_call.type() = empty_typet();
  reverse_call.location() = converter_.get_location_from_decl(call_);

  // Reverse the compile-time type-info vector to mirror the runtime
  // reordering, so that subsequent index-based type lookups remain valid.
  python_list::reverse_type_info(list_symbol->id.as_string());

  return reverse_call;
}

bool function_call_expr::is_set_method_call() const
{
  if (call_["func"]["_type"] != "Attribute")
    return false;

  const std::string &method_name = function_id_.get_function();
  if (
    method_name != "add" && method_name != "discard" &&
    method_name != "issubset" && method_name != "issuperset" &&
    method_name != "isdisjoint" && method_name != "update" &&
    method_name != "symmetric_difference")
    return false;

  // set()/frozenset() constructor receivers (e.g. set(x).issuperset(y)) are
  // sets by construction. Decide from the AST: resolving a Call receiver
  // through get_object_list_symbol() would emit the set-construction IR as a
  // side-effect, and discriminators must stay pure.
  const auto &func_value = call_["func"]["value"];
  if (func_value["_type"] == "Call")
    return func_value["func"].contains("id") &&
           (func_value["func"]["id"] == "set" ||
            func_value["func"]["id"] == "frozenset");

  std::string dummy;
  const symbolt *sym = get_object_list_symbol(dummy);
  return sym != nullptr && sym->is_set;
}

exprt function_call_expr::handle_set_method() const
{
  const std::string &method_name = function_id_.get_function();
  const auto &args = call_["args"];

  if (args.size() != 1)
    throw std::runtime_error(method_name + "() takes exactly one argument");

  // set(<iterable>).issubset/issuperset/isdisjoint(y): set() here only
  // deduplicates, which cannot change a subset/superset/disjoint verdict. Use
  // the iterable directly and skip materializing the set — a guard like
  // `set(xs).issuperset(...)` inside a loop would otherwise rebuild the set
  // (one push plus one containment scan per element) on every iteration
  // (#4805).
  if (
    method_name == "issubset" || method_name == "issuperset" ||
    method_name == "isdisjoint")
  {
    const auto &receiver = call_["func"]["value"];
    if (
      receiver["_type"] == "Call" && receiver["func"].contains("id") &&
      (receiver["func"]["id"] == "set" ||
       receiver["func"]["id"] == "frozenset") &&
      receiver["args"].size() == 1)
    {
      exprt iterable = converter_.get_expr(receiver["args"][0]);
      if (iterable.type() == type_handler_.get_list_type())
      {
        exprt other = converter_.get_expr(args[0]);
        python_set set_helper(converter_, call_);
        return set_helper.build_set_relation_call(
          iterable, other, call_, method_name);
      }
    }
  }

  std::string set_display_name;
  const symbolt *set_symbol = get_object_list_symbol(set_display_name);
  materialize_list_symbol(set_symbol);
  if (!set_symbol)
    throw std::runtime_error("Set variable not found: " + set_display_name);

  // add/discard take a single element value.
  if (method_name == "add" || method_name == "discard")
  {
    exprt elem = converter_.get_expr(args[0]);
    python_list helper(converter_, call_);
    return helper.build_set_membership_call(
      *set_symbol, call_, elem, method_name);
  }

  // issubset / issuperset / isdisjoint / update / symmetric_difference take
  // another set/iterable.
  exprt other = converter_.get_expr(args[0]);
  python_set set_helper(converter_, call_);
  return set_helper.build_set_method_call(
    *set_symbol, other, call_, method_name);
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
    method_name != "copy" && method_name != "sort" &&
    method_name != "reverse" && method_name != "popleft" &&
    method_name != "appendleft" && method_name != "count" &&
    method_name != "index")
    return false;

  // Tuples are claimed earlier; only claim count/index here when the receiver
  // resolves to a list symbol so str receivers fall through.
  if (method_name == "count" || method_name == "index")
  {
    std::string dummy;
    const symbolt *sym = get_object_list_symbol(dummy);
    const typet list_type = type_handler_.get_list_type();
    return sym != nullptr && sym->get_type() == list_type;
  }

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
    return sym != nullptr && sym->get_type() == list_type;
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
    return sym != nullptr && sym->get_type() == list_type;
  }

  // "clear" is shared between list and dict. Treat as list.clear() only when
  // the receiver resolves to a list symbol; otherwise fall through to
  // handle_dict_method(). Without this guard a dict receiver was claimed by
  // the catch-all below and passed to __ESBMC_list_clear, which dereferenced
  // the dict struct as a PyListObject and reported a spurious out-of-bounds
  // (VERIFICATION FAILED).
  if (method_name == "clear")
  {
    if (
      call_["func"].contains("value") &&
      call_["func"]["value"].contains("_type") &&
      call_["func"]["value"]["_type"] == "BinOp")
      return true;

    std::string dummy;
    const symbolt *sym = get_object_list_symbol(dummy);
    const typet list_type = type_handler_.get_list_type();
    return sym != nullptr && sym->get_type() == list_type;
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
  if (method_name == "popleft")
    return handle_list_popleft();
  if (method_name == "appendleft")
    return handle_list_appendleft();
  if (method_name == "copy")
    return handle_list_copy();
  if (method_name == "remove")
    return handle_list_remove();
  if (method_name == "sort")
    return handle_list_sort();
  if (method_name == "reverse")
    return handle_list_reverse();
  if (method_name == "count")
    return handle_list_count();
  if (method_name == "index")
    return handle_list_index();
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

    code_declt tmp_decl(build_symbol(tmp_var));
    tmp_decl.location() = converter_.get_location_from_decl(call_);
    converter_.current_block->copy_to_operands(tmp_decl);

    // Create function call with lhs
    code_function_callt new_call;
    new_call.function() = func_expr;
    new_call.arguments() = func_args;
    new_call.lhs() = build_symbol(tmp_var);
    new_call.type() = ret_type;
    new_call.location() = converter_.get_location_from_decl(call_);
    converter_.current_block->copy_to_operands(new_call);

    // Replace value_to_append with the temporary variable
    value_to_append = build_symbol(tmp_var);
  }

  // Treat a single-element char array (`char[1]`) as a `char *` for storage
  // semantics — list elements of string type are stored as pointer values
  // (see build_push_list_call). The in-place type rewrite is safe for a
  // symbol-expr value (a real variable whose storage decays to char*), but
  // would corrupt a constant char[1] expression: rewriting the type produces
  // a malformed `constant{type=char *, op0=char '\0'}` that crashes
  // migrate_expr at GOTO-program generation (#4807). Keep constants on the
  // generic array path; build_push_list_call then takes their address.
  if (
    !value_to_append.is_constant() && value_to_append.type().is_array() &&
    value_to_append.type().subtype() == char_type())
  {
    const array_typet &array_type = to_array_type(value_to_append.type());
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
    code_declt append_value(build_symbol(append_value_symbol));
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

bool function_call_expr::is_all_call() const
{
  const std::string &func_name = function_id_.get_function();
  return func_name == "all";
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

    // Set methods (add, discard) — matched before list methods so set
    // receivers don't fall through to the list handler.
    {[this]() { return is_set_method_call(); },
     [this]() { return handle_set_method(); },
     "set methods"},

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
         // V.3: build isnan in IREP2.
         expr2tc arg2;
         migrate_expr(arg_expr, arg2);
         return migrate_expr_back(isnan2tc(arg2));
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
         // V.3: build isinf in IREP2.
         expr2tc arg2;
         migrate_expr(arg_expr, arg2);
         return migrate_expr_back(isinf2tc(arg2));
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
             model_call.function() = build_symbol(*model_sym);
             model_call.arguments() = {z};
             model_call.type() =
               to_code_type(model_sym->get_type()).return_type();
             model_call.location() = converter_.get_location_from_decl(call_);
             return model_call;
           }
         }
       }

       return converter_.get_complex_handler().handle_cmath_log(
         func_name, call_, args, keywords);
     },
     "cmath log/log10"},

    // cmath inverse functions: on pure-imaginary inputs they have an exact
    // closed form, so use a fast path there and delegate every other input to
    // the Python cmath model. asin/atan/asinh/atanh are purely imaginary on
    // the imaginary axis; acos/acosh keep a nonzero real part.
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
         func_name == "atanh" || func_name == "acos" || func_name == "acosh");
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
       model_call.function() = build_symbol(*model_symbol);
       model_call.arguments() = {z};
       model_call.type() = to_code_type(model_symbol->get_type()).return_type();
       model_call.location() = converter_.get_location_from_decl(call_);

       python_math &math = converter_.get_math_handler();
       exprt zr = build_member(z, "real", double_type());
       exprt zi = build_member(z, "imag", double_type());
       exprt zero = from_double(0.0, double_type());
       // V.3: build the pure-imaginary guard (zr == 0) in IREP2.
       expr2tc zr2, zero2;
       migrate_expr(zr, zr2);
       migrate_expr(zero, zero2);
       expr2tc fast_guard = equality2tc(zr2, zero2);

       // acos(i*y) and acosh(i*y) have a nonzero real part, but a closed form
       // valid for every real y, so their fast path covers all pure-imaginary
       // inputs (zr == 0):
       //   acos(i*y)  = (pi/2, -asinh(y))
       //   acosh(i*y) = (asinh(|y|), copysign(pi/2, y))
       if (func_name == "acos" || func_name == "acosh")
       {
         // pi/2 at double precision (CPython's exact value); the fast path is
         // CPython-faithful and supersedes the model on the imaginary axis.
         exprt half_pi = from_double(1.5707963267948966, double_type());
         exprt fast_path;
         if (func_name == "acos")
         {
           exprt asinh_zi = math.handle_asinh(zi, call_);
           if (is_cpp_throw_expr(asinh_zi))
             return asinh_zi;
           expr2tc asinh_zi2;
           migrate_expr(asinh_zi, asinh_zi2);
           exprt neg_asinh =
             migrate_expr_back(neg2tc(migrate_type(double_type()), asinh_zi2));
           fast_path = make_complex(half_pi, neg_asinh);
         }
         else
         {
           exprt abs_zi = math.handle_fabs(zi, call_);
           if (is_cpp_throw_expr(abs_zi))
             return abs_zi;
           exprt real_part = math.handle_asinh(abs_zi, call_);
           if (is_cpp_throw_expr(real_part))
             return real_part;
           exprt imag_part = math.handle_copysign(half_pi, zi, call_);
           if (is_cpp_throw_expr(imag_part))
             return imag_part;
           fast_path = make_complex(real_part, imag_part);
         }
         return if_exprt(migrate_expr_back(fast_guard), fast_path, model_call);
       }

       // asin/atan/asinh/atanh map the imaginary axis onto itself, so their
       // fast path has a zero real part.
       exprt imag_result;
       if (func_name == "asin")
         imag_result = math.handle_asinh(zi, call_);
       else if (func_name == "atan")
         imag_result = math.handle_atanh(zi, call_);
       else if (func_name == "asinh")
         imag_result = math.handle_asin(zi, call_);
       else
         imag_result = math.handle_atan(zi, call_);

       if (is_cpp_throw_expr(imag_result))
         return imag_result;

       exprt fast_path = make_complex(zero, imag_result);

       // For atan(i*y) and asinh(i*y), the pure-imag shortcut only matches
       // the principal branch safely within the unit interval.
       if (func_name == "atan" || func_name == "asinh")
       {
         exprt abs_zi = math.handle_fabs(zi, call_);
         if (is_cpp_throw_expr(abs_zi))
           return abs_zi;

         exprt one = from_double(1.0, double_type());
         expr2tc abs_zi2, one2;
         migrate_expr(abs_zi, abs_zi2);
         migrate_expr(one, one2);
         expr2tc imag_guard = func_name == "atan"
                                ? lessthan2tc(abs_zi2, one2)
                                : lessthanequal2tc(abs_zi2, one2);
         fast_guard = and2tc(fast_guard, imag_guard);
       }

       return if_exprt(migrate_expr_back(fast_guard), fast_path, model_call);
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
         // Domain check for sqrt: operand must be >= 0.
         exprt domain_check;
         if (arg_expr.type().is_pointer())
         {
           // A pointer-typed ("any") operand -- e.g. an unannotated parameter
           // bound to a dynamic-list element (#2848) -- has no numeric value,
           // so typecasting it to float builds an FP op over a pointer sort
           // that aborts the SMT backend (get_significand_width; see
           // humaneval/39). Its sign is unknown, so guard the domain error
           // with a nondet condition: both the math-domain-error path and the
           // normal path stay reachable (sound), while handle_sqrt below
           // over-approximates the result as nondet.
           domain_check =
             side_effect_expr_nondett(type_handler_.get_typet("bool", 0));
         }
         else
         {
           // V.3: build the "operand < 0" guard in IREP2.
           expr2tc double_operand;
           migrate_expr(arg_expr, double_operand);
           if (!arg_expr.type().is_floatbv())
             double_operand = typecast2tc(
               migrate_type(type_handler_.get_typet("float", 0)),
               double_operand);

           domain_check = migrate_expr_back(
             lessthan2tc(double_operand, gen_zero(double_operand->type)));
         }

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

         // Now compute sqrt (>= 0 enforced above for numeric operands).
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
         // Domain check for log: operand must be > 0 (V.3: built in IREP2).
         expr2tc fp_operand;
         migrate_expr(arg_expr, fp_operand);
         if (!arg_expr.type().is_floatbv())
           fp_operand = typecast2tc(
             migrate_type(type_handler_.get_typet("float", 0)), fp_operand);
         exprt domain_check = migrate_expr_back(
           lessthanequal2tc(fp_operand, gen_zero(fp_operand->type)));
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
         // (V.3: built in IREP2).
         const type2tc float_type2 =
           migrate_type(type_handler_.get_typet("float", 0));
         expr2tc double_operand;
         migrate_expr(arg_expr, double_operand);
         if (!arg_expr.type().is_floatbv())
           double_operand = typecast2tc(float_type2, double_operand);

         expr2tc pos_one = gen_one(float_type2);
         expr2tc neg_one = neg2tc(float_type2, pos_one);

         exprt domain_check = migrate_expr_back(or2tc(
           lessthan2tc(double_operand, neg_one),
           greaterthan2tc(double_operand, pos_one)));

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
               code_declt decl(build_symbol(tmp));
               decl.location() = converter_.get_location_from_decl(call_);
               converter_.current_block->copy_to_operands(decl);
               arg = build_symbol(tmp);
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

    // pow() builtin (2- and 3-argument forms)
    {[this]() {
       return function_id_.get_function() == "pow" &&
              function_id_.get_prefix() == "py:";
     },
     [this]() { return handle_pow(); },
     "pow() builtin"},

    // issubclass() builtin
    {[this]() {
       return function_id_.get_function() == "issubclass" &&
              function_id_.get_prefix() == "py:";
     },
     [this]() { return handle_issubclass(); },
     "issubclass() builtin"},

    // callable() builtin
    {[this]() {
       return function_id_.get_function() == "callable" &&
              function_id_.get_prefix() == "py:";
     },
     [this]() { return handle_callable(); },
     "callable() builtin"},

    // ascii() builtin
    {[this]() {
       return function_id_.get_function() == "ascii" &&
              function_id_.get_prefix() == "py:";
     },
     [this]() { return handle_ascii(); },
     "ascii() builtin"},

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
              !elem_sym || !elem_sym->get_value().is_constant() ||
              !(elem_sym->get_type().is_signedbv() ||
                elem_sym->get_type().is_unsignedbv()))
            {
              all_constant_ints = false;
              break;
            }

            BigInt key =
              binary2integer(elem_sym->get_value().value().c_str(), true);
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

          // Concrete tuple path: a literal list of constant integer tuples
          // (e.g. sorted([(3,1),(1,2)])). The runtime tuple-sort model retypes
          // elements as int; sort here at convert time and rebuild a list of
          // tuple literals so the element type is preserved and verification is
          // cheap. Symbolic tuple lists fall through (still unsupported).
          std::function<bool(const exprt &, BigInt &)> eval_const_int =
            [&](const exprt &e, BigInt &out) -> bool {
            if (
              e.is_constant() && (e.type().is_signedbv() ||
                                  e.type().is_unsignedbv() || e.is_boolean()))
            {
              out = binary2integer(
                to_constant_expr(e).value().c_str(), e.type().is_signedbv());
              return true;
            }
            if (e.is_symbol())
            {
              const symbolt *s =
                converter_.find_symbol(e.identifier().as_string());
              return s && eval_const_int(s->get_value(), out);
            }
            // A negative literal reaches here as unary-minus over a constant
            // (the parser emits UnaryOp(USub, Constant(n))); a widened literal
            // as a typecast. Fold both.
            if (e.id() == "unary-" && e.operands().size() == 1)
            {
              if (!eval_const_int(e.op0(), out))
                return false;
              out = -out;
              return true;
            }
            if (e.id() == "typecast" && e.operands().size() == 1)
              return eval_const_int(e.op0(), out);
            return false;
          };

          struct sortable_tuple
          {
            std::vector<BigInt> key;
            size_t pos;
          };
          std::vector<sortable_tuple> telems;
          bool all_constant_tuples = true;
          size_t arity = 0;

          for (size_t i = 0; i < map_size && all_constant_tuples; ++i)
          {
            const std::string elem_id =
              python_list::get_list_element_id(list_id, i);
            const symbolt *elem_sym =
              elem_id.empty() ? nullptr : converter_.find_symbol(elem_id);
            exprt val = elem_sym ? elem_sym->get_value() : exprt();
            while (val.is_symbol())
            {
              const symbolt *s =
                converter_.find_symbol(val.identifier().as_string());
              if (!s)
                break;
              val = s->get_value();
            }
            if (
              !elem_sym ||
              !converter_.get_tuple_handler().is_tuple_type(
                elem_sym->get_type()) ||
              val.id() != "struct" || val.operands().empty())
            {
              all_constant_tuples = false;
              break;
            }
            if (i == 0)
              arity = val.operands().size();
            else if (val.operands().size() != arity)
            {
              all_constant_tuples = false;
              break;
            }
            std::vector<BigInt> key;
            for (const auto &comp : val.operands())
            {
              BigInt v;
              if (!eval_const_int(comp, v))
              {
                all_constant_tuples = false;
                break;
              }
              key.push_back(v);
            }
            if (!all_constant_tuples)
              break;
            telems.push_back({std::move(key), i});
          }

          if (all_constant_tuples && !telems.empty())
          {
            std::stable_sort(
              telems.begin(),
              telems.end(),
              [](const sortable_tuple &a, const sortable_tuple &b) {
                if (a.key == b.key)
                  return a.pos < b.pos;
                return a.key < b.key; // lexicographic on the component vector
              });
            if (fast_path_reverse)
              std::reverse(telems.begin(), telems.end());

            nlohmann::json sorted_list;
            sorted_list["_type"] = "List";
            sorted_list["elts"] = nlohmann::json::array();
            converter_.copy_location_fields_from_decl(call_, sorted_list);
            for (const auto &te : telems)
            {
              nlohmann::json tup;
              tup["_type"] = "Tuple";
              tup["elts"] = nlohmann::json::array();
              converter_.copy_location_fields_from_decl(call_, tup);
              for (const BigInt &v : te.key)
              {
                // Mirror the parser's literal shape: a negative integer is
                // UnaryOp(USub, Constant(|v|)), not Constant(-v). A bare
                // negative Constant nested in a tuple takes a slow conversion
                // path, so emit the UnaryOp form for negatives.
                nlohmann::json cst;
                cst["_type"] = "Constant";
                cst["value"] = (v < 0 ? -v : v).to_int64();
                cst["kind"] = nullptr;
                converter_.copy_location_fields_from_decl(call_, cst);
                if (v < 0)
                {
                  nlohmann::json neg;
                  neg["_type"] = "UnaryOp";
                  neg["op"] = {{"_type", "USub"}};
                  neg["operand"] = cst;
                  converter_.copy_location_fields_from_decl(call_, neg);
                  tup["elts"].push_back(neg);
                }
                else
                  tup["elts"].push_back(cst);
              }
              sorted_list["elts"].push_back(tup);
            }

            python_list sorted_list_expr(converter_, sorted_list);
            return sorted_list_expr.get();
          }

          // Symbolic tuple path: a list of tuples whose components may be
          // symbolic (e.g. sorted([(a, b), (b, a)])). The runtime sort model
          // compares the tuple storage as reinterpreted integers — not
          // Python's lexicographic order — and retypes the result elements as
          // int. Instead emit a convert-time oblivious sorting network
          // (selection sort) that compares tuples lexicographically and
          // selects elements with ite, producing a correctly ordered list
          // whose elements keep their tuple type. Bounded to a small length to
          // keep the ite trees manageable.
          if (map_size <= 16)
          {
            std::vector<exprt> vals;
            vals.reserve(map_size);
            bool all_tuples = true;
            typet tuple_type;
            for (size_t i = 0; i < map_size; ++i)
            {
              const std::string elem_id =
                python_list::get_list_element_id(list_id, i);
              const symbolt *elem_sym =
                elem_id.empty() ? nullptr : converter_.find_symbol(elem_id);
              if (
                !elem_sym || !converter_.get_tuple_handler().is_tuple_type(
                               elem_sym->get_type()))
              {
                all_tuples = false;
                break;
              }
              // Heterogeneous tuples (different arity/component types) are left
              // to the model; a homogeneous list is the sortable case.
              if (i == 0)
                tuple_type = elem_sym->get_type();
              else if (elem_sym->get_type() != tuple_type)
              {
                all_tuples = false;
                break;
              }
              vals.push_back(build_symbol(*elem_sym));
            }

            if (all_tuples && vals.size() == map_size && map_size > 0)
            {
              const locationt loc = converter_.get_location_from_decl(call_);
              // Materialize a value into a fresh temp symbol so later
              // comparisons reference the symbol rather than a nested ite tree.
              // exprt has value semantics (no subexpression sharing), so
              // threading raw ite trees through the network would blow up
              // exponentially in the number of compare-exchanges.
              auto materialize = [&](const exprt &value) -> exprt {
                symbolt &tmp = converter_.create_tmp_symbol(
                  call_, "$sort_tmp$", value.type(), value);
                code_declt decl(build_symbol(tmp));
                decl.copy_to_operands(value);
                decl.location() = loc;
                converter_.add_instruction(decl);
                return build_symbol(tmp);
              };

              // Each compare-exchange puts the smaller (or larger, when
              // reverse=True) tuple at the earlier slot. This selection-sort
              // network is not stable, but stability is moot here: the
              // comparison is over the whole tuple, so two elements that
              // compare equal are bit-identical and reordering them is
              // unobservable. (Do not reuse this network for a partial-key
              // sort, where ties would be distinguishable.)
              const std::string cmp_op = fast_path_reverse ? "Gt" : "Lt";
              bool network_ok = true;
              for (size_t i = 0; i < vals.size() && network_ok; ++i)
              {
                for (size_t j = i + 1; j < vals.size(); ++j)
                {
                  exprt a = vals[j];
                  exprt b = vals[i];
                  exprt cond =
                    converter_.handle_tuple_operations(cmp_op, a, b, call_);
                  if (cond.is_nil() || !cond.type().is_bool())
                  {
                    network_ok = false; // non-orderable components
                    break;
                  }
                  exprt old_i = vals[i];
                  exprt old_j = vals[j];
                  vals[i] = materialize(if_exprt(cond, old_j, old_i));
                  vals[j] = materialize(if_exprt(cond, old_i, old_j));
                }
              }

              if (network_ok)
              {
                python_list result(converter_, call_);
                return result.build_list_from_exprs(vals);
              }
            }
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
  const bool is_sorted_min_max = func_name == "min" || func_name == "max" ||
                                 func_name == "sorted" ||
                                 func_name == "reversed";

  // min(iter, default=...) / max(iter, default=...) route to *_default
  // variants that fall back to the supplied default when iter is empty
  // instead of raising ValueError.
  bool has_default_kwarg = false;
  exprt default_kwarg_value;
  if ((func_name == "min" || func_name == "max") && call_.contains("keywords"))
  {
    for (const auto &kw : call_["keywords"])
    {
      if (kw.value("arg", "") == "default")
      {
        has_default_kwarg = true;
        default_kwarg_value = converter_.get_expr(kw["value"]);
        break;
      }
    }
  }

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
        python_list::has_mixed_numeric_types(list_id) && !has_default_kwarg)
      {
        irep_idt comparison_op =
          (func_name == "max") ? exprt::i_gt : exprt::i_lt;
        python_list list_helper(converter_, call_["args"][0]);
        return list_helper.build_min_max_for_mixed_numeric(
          list_arg, list_id, func_name, comparison_op);
      }
    }
    // Dispatch to typed builtin based on element type
    if (has_default_kwarg)
      actual_func_name += "_default";
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
    if (var_symbol && !var_symbol->get_type().is_code())
    {
      side_effect_expr_function_callt call;
      call.location() = converter_.get_location_from_decl(call_);
      exprt func_expr = build_symbol(*var_symbol);
      if (
        !var_symbol->get_type().is_pointer() ||
        !var_symbol->get_type().subtype().is_code())
        func_expr = build_typecast(func_expr, gen_pointer_type(code_typet()));
      call.function() = func_expr;

      bool resolved = false;
      if (
        var_symbol->get_value().is_address_of() &&
        !var_symbol->get_value().operands().empty() &&
        var_symbol->get_value().op0().is_symbol())
      {
        const symbolt *target_symbol =
          symbol_table.find_symbol(var_symbol->get_value().op0().identifier());
        if (target_symbol && target_symbol->get_type().is_code())
        {
          call.type() = to_code_type(target_symbol->get_type()).return_type();
          resolved = true;
        }
      }
      if (
        !resolved && var_symbol->get_type().is_pointer() &&
        var_symbol->get_type().subtype().is_code())
      {
        call.type() =
          to_code_type(var_symbol->get_type().subtype()).return_type();
        resolved = true;
      }
      if (!resolved)
        call.type() = any_type();

      for (const auto &arg_node : call_["args"])
      {
        exprt arg = converter_.get_expr(arg_node);
        if (arg.type().is_code() && arg.is_symbol())
          arg = build_address_of(arg);
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
          call.arguments().push_back(build_address_of(arg));
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
              // V.3: build the void* null fallback via the IREP2 factory.
              exprt zero_fallback =
                migrate_expr_back(gen_zero(migrate_type(any_type())));
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
            exprt receiver = build_symbol(*obj_symbol);
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
                code_declt tmp_decl(build_symbol(tmp));
                tmp_decl.location() = location;
                converter_.current_block->copy_to_operands(tmp_decl);
                arg = build_symbol(tmp);
              }
              call.arguments().push_back(build_address_of(arg));
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
                code_declt tmp_decl(build_symbol(tmp));
                tmp_decl.location() = location;
                converter_.current_block->copy_to_operands(tmp_decl);
                arg = build_symbol(tmp);
              }
              call.arguments().push_back(build_address_of(arg));
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
          // Also add an assertion to the current block to flag this as an
          // error (V.3: build the always-fail condition in IREP2).
          code_assertt assert_code(migrate_expr_back(gen_false_expr()));
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
  call.function() = build_symbol(*func_symbol);
  const typet &return_type =
    to_code_type(func_symbol->get_type()).return_type();
  call.type() = return_type;

  auto bind_instance_receiver = [&](exprt receiver) -> exprt {
    return receiver.type().is_pointer() ? receiver : gen_address_of(receiver);
  };

  auto bind_instance_receiver_symbol =
    [&](const symbolt &receiver_symbol) -> exprt {
    exprt receiver = build_symbol(receiver_symbol);
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
      // Stage 1 object-model migration (#3067/#4773): when the LHS has been
      // typed as a pointer-to-class reference, allocate the instance as a
      // typed, non-expiring object and pass the pointer itself as `self`, so
      // the object survives escaping its defining function. Otherwise keep the
      // legacy in-place struct construction (self = &lhs). `__ESBMC_new_object`
      // is intercepted in symex (symex_mem_inf): the LHS pointer type carries
      // the class type, so the object is sized symbolically by the struct at
      // symex time — robust to the class struct still gaining fields after this
      // construction (which a byte-sized allocation cannot handle).
      // If the lvalue is `object`/Any (a pointer to void/empty), the
      // new_object interception cannot size the allocation — the pointee type
      // has no width. Retype the lvalue (and its symbol) to the class being
      // constructed; a `void*`/Any slot legitimately holds the resulting
      // `Class*` pointer. Without this, `t: object = Box()` aborts in symex.
      if (
        converter_.current_lhs->type().is_pointer() &&
        (converter_.current_lhs->type().subtype().id() == "empty" ||
         converter_.current_lhs->type().subtype().id().empty()))
      {
        const typet class_ptr = gen_pointer_type(call.type());
        converter_.current_lhs->type() = class_ptr;
        if (converter_.current_lhs->is_symbol())
          if (
            symbolt *s = converter_.symbol_table().find_symbol(
              converter_.current_lhs->identifier()))
            s->set_type(class_ptr);
      }
      if (converter_.current_lhs->type().is_pointer())
      {
        const symbolt *new_obj_sym =
          converter_.symbol_table().find_symbol("c:@F@__ESBMC_new_object");
        assert(new_obj_sym && "__ESBMC_new_object model required");
        code_function_callt alloc_call;
        alloc_call.lhs() = *converter_.current_lhs;
        alloc_call.function() = symbol_expr(*new_obj_sym);
        converter_.add_instruction(alloc_call);
        call.arguments().push_back(*converter_.current_lhs);
      }
      else
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
          code_declt tmp_decl(build_symbol(tmp));
          tmp_decl.location() = location;
          converter_.current_block->copy_to_operands(tmp_decl);

          // Call the constructor if it is defined, using tmp as self.
          exprt *saved_lhs = converter_.current_lhs;
          exprt tmp_expr = build_symbol(tmp);
          converter_.current_lhs = &tmp_expr;
          exprt ctor_result = converter_.get_expr(func_value);
          converter_.current_lhs = saved_lhs;

          call.arguments().push_back(bind_instance_receiver(build_symbol(tmp)));
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
            code_declt tmp_decl(build_symbol(tmp));
            tmp_decl.location() = location;
            converter_.current_block->copy_to_operands(tmp_decl);

            // Process the inner call; set its LHS to tmp so the return value
            // is stored there (emits: FUNCTION_CALL: tmp = inner_call(...)).
            exprt inner_call = converter_.get_expr(func_value);
            if (
              inner_call.is_code() && inner_call.statement() == "function_call")
            {
              inner_call.op0() = build_symbol(tmp);
              inner_call.location() = location;
              converter_.add_instruction(inner_call);
            }
            call.arguments().push_back(
              bind_instance_receiver(build_symbol(tmp)));
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
    const code_typet &func_type = to_code_type(func_symbol->get_type());
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
      // (V.3: build the void* null via the IREP2 factory).
      typet t = pointer_typet(empty_typet());
      call.arguments().push_back(migrate_expr_back(gen_zero(migrate_type(t))));
      param_offset = 1;

      // All methods for the int/float classes without parameters act solely
      // on the encapsulated scalar value. Therefore, we always pass the caller
      // (obj) as a parameter in these functions. For example, if x is an int
      // instance, x.bit_length() call becomes bit_length(x); likewise a float
      // instance's x.is_integer() becomes is_integer(x).
      const std::string recv_type =
        obj_symbol ? type_handler_.get_var_type(obj_symbol->name.as_string())
                   : std::string();
      if (
        obj_symbol && call_["args"].empty() &&
        (recv_type == "int" || recv_type == "float"))
      {
        call.arguments().push_back(build_symbol(*obj_symbol));
      }
      else if (call_["func"]["value"]["_type"] == "BinOp")
      {
        // Handling function call from binary expressions such as: (x+1).bit_length()
        call.arguments().push_back(converter_.get_expr(call_["func"]["value"]));
      }
    }
  }

  // Get function type and parameters for Optional wrapping
  const code_typet &func_type = to_code_type(func_symbol->get_type());
  const auto &params = func_type.arguments();

  size_t arg_index = 0;
  for (const auto &arg_node : call_["args"])
  {
    // An argument expression does not bind to the outer assignment's LHS.
    // Clearing `current_lhs` while evaluating each argument prevents inner
    // constructor calls (e.g. `f(A())`) from using an unrelated LHS as their
    // `self` storage; they will allocate a `$ctor_self$` temp instead
    // (GitHub #4552).
    exprt *saved_lhs = converter_.current_lhs;
    converter_.current_lhs = nullptr;
    exprt arg = converter_.get_expr(arg_node);
    converter_.current_lhs = saved_lhs;

    // A function name passed as an argument decays to a function pointer,
    // mirroring C's implicit function-to-pointer conversion.
    if (arg.type().is_code() && arg.is_symbol())
      arg = build_address_of(arg);

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
        code_declt temp_decl(build_symbol(temp_symbol));
        temp_decl.location() = location;
        converter_.current_block->copy_to_operands(temp_decl);

        // Assign the character to the first element
        exprt temp_array = build_symbol(temp_symbol);
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
        arg = build_address_of(build_symbol(temp_symbol));
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
          code_declt tmp_decl(build_symbol(tmp));
          tmp_decl.location() = location;
          converter_.current_block->copy_to_operands(tmp_decl);
          code_assignt tmp_assign(build_symbol(tmp), arg);
          tmp_assign.location() = location;
          converter_.current_block->copy_to_operands(tmp_assign);
          arg = build_symbol(tmp);
        }
        arg = build_typecast(build_address_of(arg), param_type);
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
          code_declt tmp_decl(build_symbol(tmp));
          tmp_decl.location() = location;
          converter_.current_block->copy_to_operands(tmp_decl);
          code_assignt tmp_assign(build_symbol(tmp), arg);
          tmp_assign.location() = location;
          converter_.current_block->copy_to_operands(tmp_assign);
          arg = build_symbol(tmp);
        }

        arg = build_address_of(arg);
        if (!base_type_eq(arg.type(), param_type, converter_.ns))
          arg = build_typecast(arg, param_type);
      }
    }

    // Handle string literal constants
    // Ensure they are proper null-terminated arrays.
    // Guard: skip complex literals whose JSON "value" field is a string
    // representation of the complex number (e.g. "0.5j") — get_expr already
    // returned the correct complex struct via get_literal's annotation check.
    // Overwriting it with a string literal here would produce an
    // address-of-string argument ("got pointer, expected struct" crash).
    const bool arg_is_complex_literal =
      arg_node["_type"] == "Constant" &&
      arg_node.value("esbmc_type_annotation", std::string()) == "complex";
    if (
      !arg_is_complex_literal && arg_node["_type"] == "Constant" &&
      arg_node["value"].is_string())
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
        // get_expr() may hand back an inline list-returning call -- e.g. the
        // sorted(m) in len(sorted(m)) -- as a code_function_callt statement
        // rather than a value. Assigning that statement to a temp discards the
        // call's return value, leaving the temp at its NONDET decl and giving a
        // wrong size (#5464). materialize_list_function_call() binds the return
        // value into a temp and yields that symbol (the same path the
        // sorted()[i] subscript sibling uses, #4807); non-call args pass
        // through unchanged.
        arg = converter_.materialize_list_function_call(
          arg, call_, *converter_.current_block);

        if (arg.is_symbol())
          list_symbol = converter_.find_symbol(arg.identifier().as_string());
        else
        {
          const typet list_type = type_handler_.get_list_type();
          symbolt &tmp_list = converter_.create_tmp_symbol(
            call_, "$obj_size_list_arg$", list_type, exprt());

          code_declt tmp_decl(build_symbol(tmp_list));
          tmp_decl.location() = location;
          converter_.current_block->copy_to_operands(tmp_decl);

          code_assignt tmp_assign(build_symbol(tmp_list), arg);
          tmp_assign.location() = location;
          converter_.current_block->copy_to_operands(tmp_assign);

          list_symbol = &tmp_list;
        }
      }

      assert(list_symbol);

      const symbolt *list_size_func_sym =
        converter_.find_symbol("c:@F@__ESBMC_list_size");
      assert(list_size_func_sym);

      code_function_callt list_size_func_call;
      list_size_func_call.function() = build_symbol(*list_size_func_sym);

      // passing arguments to list_size
      list_size_func_call.arguments().push_back(build_symbol(*list_symbol));

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
          const code_typet &func_type = to_code_type(func_symbol->get_type());
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
        static_cast<const code_typet &>(func_symbol->get_type());
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
        code_declt tmp_decl(build_symbol(tmp));
        tmp_decl.location() = location;
        converter_.current_block->copy_to_operands(tmp_decl);
        arg = build_symbol(tmp);
      }
      call.arguments().push_back(build_address_of(arg));
    }
    else
      call.arguments().push_back(arg);

    arg_index++;
  }

  // Forward keyword arguments to their parameter slots so the callee
  // receives the supplied value. The validation loop below only fills in
  // default values for *missing* params, so kwargs would otherwise be
  // marked "provided" yet never actually passed. Subsumes the
  // min/max-default specific path: with the *_default models exposing a
  // named `default` parameter, the generic forwarding lands the value in
  // the right slot for free.
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

      if (var_type == "int" || var_type == "float")
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
            default_val = build_address_of(default_val);
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
      code_declt temp_decl(build_symbol(temp_self));
      temp_decl.location() = location;
      converter_.current_block->copy_to_operands(temp_decl);

      // Insert self as first argument
      call.arguments().insert(
        call.arguments().begin(), gen_address_of(build_symbol(temp_self)));

      // Emit the constructor call directly as a FUNCTION_CALL instruction so
      // ESBMC inlines it (codet("expression") would produce OTHER and be
      // skipped).  Return the initialised temp_self symbol so callers such as
      // list literals receive the properly constructed object.
      call.location() = location;
      converter_.add_instruction(call);
      return build_symbol(temp_self);
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
    typet obj_type = obj_symbol->get_type();
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

  // V.3: build the always-fail assert condition in IREP2.
  code_assertt assert_code(migrate_expr_back(gen_false_expr()));
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

  const code_typet &func_type = to_code_type(func_symbol->get_type());
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
