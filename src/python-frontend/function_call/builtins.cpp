// Builtin handlers for function_call_expr.
//
// This translation unit hosts the implementations of Python builtin
// handlers (e.g., abs, round, complex, divmod, isinstance, hasattr,
// input, min/max, print, any/all, math.comb) and the helpers used
// exclusively by them. The class declaration and the remaining
// implementations live in function_call/expr.{h,cpp}.

#include <python-frontend/function_call/expr.h>
#include <python-frontend/complex_handler.h>
#include <python-frontend/complex_handler_utils.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/python_exception_handler.h>
#include <python-frontend/string/string_handler_utils.h>
#include <python-frontend/tuple_handler.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/python_expr_builder.h>
#include <util/arith_tools.h>
#include <util/base_type.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/python_types.h>
#include <util/std_expr.h>
#include <util/string_constant.h>
#include <irep2/irep2_utils.h>
#include <util/migrate.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <optional>
#include <stdexcept>

using namespace json_utils;
using namespace python_expr;

namespace
{
// Default length used for nondeterministic strings when the user has
// not provided an explicit value via --nondet-str-length.
constexpr int DEFAULT_NONDET_STR_LENGTH = 16;

int get_nondet_str_length()
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

// Banker's rounding (IEEE 754 round-half-to-even): rounds to the nearest
// integer, breaking ties toward the even neighbor. The tie_eps tolerates
// small floating-point error around the .5 boundary so that values
// representable as exact halves are detected reliably.
template <typename F>
F round_ties_to_even(const F value, const F tie_eps)
{
  const F lower = std::floor(value);
  const F diff = value - lower;

  if (diff < F(0.5) - tie_eps)
    return lower;
  if (diff > F(0.5) + tie_eps)
    return lower + F(1);

  const F parity = std::fmod(std::fabs(lower), F(2));
  const bool lower_is_even =
    parity < tie_eps || std::fabs(parity - F(2)) < tie_eps;
  return lower_is_even ? lower : lower + F(1);
}

constexpr double ROUND_TIE_EPS_DOUBLE = 1e-12;
constexpr long double ROUND_TIE_EPS_LONG_DOUBLE = 1e-15L;

double round_to_ndigits_ties_even(const double value, const int ndigits)
{
  // Keep scaling deterministic across libm implementations.
  long double scale = 1.0L;
  const int n = ndigits >= 0 ? ndigits : -ndigits;
  for (int i = 0; i < n; ++i)
    scale *= 10.0L;

  const long double scaled = ndigits >= 0
                               ? static_cast<long double>(value) * scale
                               : static_cast<long double>(value) / scale;
  const long double rounded =
    round_ties_to_even<long double>(scaled, ROUND_TIE_EPS_LONG_DOUBLE);
  return static_cast<double>(ndigits >= 0 ? rounded / scale : rounded * scale);
}

// Check if an AST node is a known-empty literal (falsy in Python).
// Needed because ESBMC's IR represents empty containers as non-NULL
// pointers/structs, making them appear truthy at the IR level.
bool is_empty_literal(const nlohmann::json &node)
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
} // namespace

exprt function_call_expr::combine_truthiness(exprt acc, exprt next, ReduceOp op)
{
  // V.3: build the any/all reduction fold in IREP2.
  expr2tc a2, n2;
  migrate_expr(acc, a2);
  migrate_expr(next, n2);
  return migrate_expr_back(
    op == ReduceOp::Any ? or2tc(a2, n2) : and2tc(a2, n2));
}

exprt function_call_expr::handle_input() const
{
  // input() returns a non-deterministic string
  // Model as a bounded C-string without embedded nulls.
  int max_str_length = get_nondet_str_length();
  typet string_type = type_handler_.get_typet("str", max_str_length);

  symbolt &input_sym =
    converter_.create_tmp_symbol(call_, "$input_str$", string_type, exprt());
  code_declt decl(build_symbol(input_sym));
  decl.location() = converter_.get_location_from_decl(call_);
  converter_.add_instruction(decl);

  exprt nondet_value("sideeffect", string_type);
  nondet_value.statement("nondet");
  code_assignt nondet_assign(build_symbol(input_sym), nondet_value);
  nondet_assign.location() = converter_.get_location_from_decl(call_);
  converter_.add_instruction(nondet_assign);

  symbolt &len_sym =
    converter_.create_tmp_symbol(call_, "$input_len$", size_type(), exprt());
  code_declt len_decl(build_symbol(len_sym));
  len_decl.location() = converter_.get_location_from_decl(call_);
  converter_.add_instruction(len_decl);

  exprt len_nondet("sideeffect", size_type());
  len_nondet.statement("nondet");
  code_assignt len_assign(build_symbol(len_sym), len_nondet);
  len_assign.location() = converter_.get_location_from_decl(call_);
  converter_.add_instruction(len_assign);

  // len_sym and the literal are both size_type (synthetic), so build the
  // length-bound comparison in IREP2 (V.3).
  exprt len_bound = build_less_than(
    build_symbol(len_sym), from_integer(max_str_length, size_type()));
  codet assume_len("assume");
  assume_len.copy_to_operands(len_bound);
  assume_len.location() = converter_.get_location_from_decl(call_);
  converter_.add_instruction(assume_len);

  exprt term_pos =
    build_index(build_symbol(input_sym), build_symbol(len_sym), char_type());
  code_assignt term_assign(term_pos, from_integer(0, char_type()));
  term_assign.location() = converter_.get_location_from_decl(call_);
  converter_.add_instruction(term_assign);

  // Record the companion $input_len$ symbol so len() on this string (or any
  // variable aliasing it) can return the symbolic length directly instead of
  // falling back to strlen() loop-unrolling.
  converter_.input_str_to_len_sym_[input_sym.id.as_string()] =
    len_sym.id.as_string();

  return build_symbol(input_sym);
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
    code_declt decl(build_symbol(nondet_str_symbol));
    decl.location() = converter_.get_location_from_decl(call_);
    converter_.add_instruction(decl);

    // Create nondet assignment for the array
    exprt nondet_value("sideeffect", char_array_type);
    nondet_value.statement("nondet");

    code_assignt nondet_assign(build_symbol(nondet_str_symbol), nondet_value);
    nondet_assign.location() = converter_.get_location_from_decl(call_);
    converter_.add_instruction(nondet_assign);

    // Ensure null terminator at the last position
    exprt last_index = from_integer(max_str_length - 1, size_type());
    exprt null_char = from_integer(0, char_type());

    exprt last_elem = build_index(build_symbol(nondet_str_symbol), last_index);
    code_assignt null_assign(last_elem, null_char);
    null_assign.location() = converter_.get_location_from_decl(call_);
    converter_.add_instruction(null_assign);

    // Return address of first element: &arr[0] which is char*
    exprt first_elem = build_index(
      build_symbol(nondet_str_symbol), from_integer(0, size_type()));
    return build_address_of(first_elem);
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
    if (var_symbol && var_symbol->get_value().is_constant())
    {
      const constant_exprt &const_val =
        to_constant_expr(var_symbol->get_value());
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
        t = build_symbol(*symbol);
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
      t = build_symbol(*symbol);
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

exprt function_call_expr::handle_issubclass() const
{
  const auto &args = call_["args"];
  if (args.size() != 2)
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError", "issubclass() takes exactly 2 arguments");

  // arg 1 must be a class, i.e. a Name referring to a class or builtin type.
  if (args[0]["_type"] != "Name")
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError", "issubclass() arg 1 must be a class");
  const std::string cls = args[0]["id"].get<std::string>();

  // arg 2 is a single class or a tuple of classes.
  std::vector<std::string> targets;
  const auto &info = args[1];
  auto add_name = [&targets](const nlohmann::json &n) {
    if (n["_type"] == "Name")
      targets.push_back(n["id"].get<std::string>());
  };
  if (info["_type"] == "Tuple")
    for (const auto &e : info["elts"])
      add_name(e);
  else
    add_name(info);

  if (targets.empty())
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError", "issubclass() arg 2 must be a class or tuple of classes");

  // Collect the ancestors of `cls` by walking the AST class hierarchy.
  // Every class is implicitly a subclass of object; bool subclasses int.
  const auto &ast = converter_.ast();
  std::vector<std::string> ancestors;
  std::vector<std::string> work{cls};
  auto seen = [&ancestors](const std::string &c) {
    return std::find(ancestors.begin(), ancestors.end(), c) != ancestors.end();
  };
  while (!work.empty())
  {
    std::string c = work.back();
    work.pop_back();
    if (seen(c))
      continue;
    ancestors.push_back(c);

    if (c == "bool")
      work.push_back("int");

    const auto cls_node = json_utils::find_class(ast["body"], c);
    if (!cls_node.empty() && cls_node.contains("bases"))
      for (const auto &b : cls_node["bases"])
        if (b["_type"] == "Name")
          work.push_back(b["id"].get<std::string>());
  }

  for (const std::string &t : targets)
    if (t == "object" || seen(t))
      return true_exprt();
  return false_exprt();
}

exprt function_call_expr::handle_callable() const
{
  const auto &args = call_["args"];
  if (args.size() != 1)
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError", "callable() takes exactly one argument");

  const auto &arg = args[0];
  const std::string node_type = arg["_type"];

  // Lambdas are callable.
  if (node_type == "Lambda")
    return true_exprt();

  // Literal containers and constants are never callable.
  if (
    node_type == "Constant" || node_type == "List" || node_type == "Tuple" ||
    node_type == "Dict" || node_type == "Set")
    return false_exprt();

  if (node_type == "Name")
  {
    const std::string name = arg["id"].get<std::string>();
    const auto &ast = converter_.ast();

    // Builtin type constructors / builtin functions, user classes, and
    // user functions are all callable.
    if (
      type_utils::is_builtin_type(name) || json_utils::is_class(name, ast) ||
      json_utils::search_function_in_ast(ast, name))
      return true_exprt();

    // Otherwise it is an ordinary variable: callable iff its class defines
    // __call__.
    const std::string var_type = type_handler_.get_var_type(name);
    if (
      !var_type.empty() && json_utils::is_class(var_type, ast) &&
      method_exists_in_class_hierarchy(var_type, "__call__"))
      return true_exprt();
    return false_exprt();
  }

  // Conservative default for forms we do not statically resolve.
  return false_exprt();
}

exprt function_call_expr::handle_pow() const
{
  const auto &args = call_["args"];

  if (args.size() != 2 && args.size() != 3)
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError", "pow() takes 2 or 3 arguments");

  exprt base = converter_.get_expr(args[0]);
  exprt exp = converter_.get_expr(args[1]);

  if (args.size() == 2)
    // pow(base, exp) shares the exact lowering of the ** operator, so integer,
    // float, and negative-exponent cases behave identically to base ** exp.
    return converter_.get_math_handler().handle_power(base, exp);

  // 3-argument form: pow(base, exp, mod) == (base ** exp) % mod. CPython
  // requires all three operands to be integers.
  exprt mod = converter_.get_expr(args[2]);
  if (
    base.type().is_floatbv() || exp.type().is_floatbv() ||
    mod.type().is_floatbv())
    return converter_.get_exception_handler().gen_exception_raise(
      "TypeError",
      "pow() 3rd argument not allowed unless all arguments are integers");

  // Modular exponentiation must be exact: the floating-point modulo used by
  // the % operator loses precision once base**exp exceeds 2^53, which would be
  // unsound for the large operands typical of modular arithmetic. We therefore
  // evaluate the constant-operand case exactly with BigInt and reject the
  // symbolic case rather than emit an unsound encoding.
  // Resolve an operand to an integer constant, folding a leading unary minus
  // (a negated literal such as -2 is a unary-minus expression, not a constant).
  std::function<std::optional<BigInt>(const exprt &)> as_int =
    [&](const exprt &e) -> std::optional<BigInt> {
    if (e.is_constant() && type_utils::is_integer_type(e.type()))
      return binary2integer(
        to_constant_expr(e).get_value().as_string(), e.type().is_signedbv());
    if ((e.id() == "unary-" || e.id() == "-") && e.operands().size() == 1)
      if (auto v = as_int(e.operands()[0]))
        return -*v;
    return std::nullopt;
  };
  std::optional<BigInt> bb = as_int(base), eb = as_int(exp), mb = as_int(mod);
  if (!bb || !eb || !mb)
    return converter_.get_exception_handler().gen_exception_raise(
      "NotImplementedError",
      "pow() with three arguments is only supported for constant integer "
      "operands");
  BigInt b = *bb;
  BigInt e = *eb;
  BigInt m = *mb;

  if (m == 0)
    return converter_.get_exception_handler().gen_exception_raise(
      "ValueError", "pow() 3rd argument cannot be 0");
  if (e < 0)
    // CPython computes a modular inverse here, which ESBMC does not model.
    return converter_.get_exception_handler().gen_exception_raise(
      "NotImplementedError",
      "pow() with a negative exponent and a modulus is not supported");

  // Right-to-left binary modular exponentiation over BigInt (exact).
  BigInt result = BigInt(1) % m;
  BigInt acc = b % m;
  BigInt rem = e;
  while (rem > 0)
  {
    if (rem % 2 != 0)
      result = (result * acc) % m;
    rem /= 2;
    acc = (acc * acc) % m;
  }
  // Python's result follows the sign of the modulus (floored modulo); adjust
  // the truncated BigInt remainder when the signs differ.
  if (result != 0 && (result < 0) != (m < 0))
    result += m;

  return from_integer(result, base.type());
}

exprt function_call_expr::handle_abs(nlohmann::json &arg) const
{
  // Build the IR for abs() over a fully resolved operand expression.
  // - dispatch __abs__ when the operand defines one,
  // - delegate to the complex handler when the operand is complex,
  // - otherwise emit a symbolic abs() expression preserving the type.
  auto build_abs = [this](exprt operand) -> exprt {
    exprt dunder_result = converter_.dispatch_unary_dunder_operator(
      "abs", operand, converter_.get_location_from_decl(call_));
    if (!dunder_result.is_nil())
      return dunder_result;
    if (is_complex_type(operand.type()))
      return converter_.get_complex_handler().handle_abs(operand);
    exprt abs_expr("abs", operand.type());
    abs_expr.copy_to_operands(operand);
    return abs_expr;
  };

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
      return build_abs(converter_.get_expr(arg));
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
    if (!sym)
      return converter_.get_exception_handler().gen_exception_raise(
        "NameError", "variable '" + var_name + "' is not defined");
    return build_abs(converter_.get_expr(arg));
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
  // Skip __abs__ dispatch in this fallback: arg_type is restricted to the
  // built-in numeric types (int/float/complex) which never define a Python
  // __abs__ method in user code, and probing it can perturb downstream SMT
  // encoding on complex operands.
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
        arg["value"] = static_cast<int>(
          round_ties_to_even<double>(val, ROUND_TIE_EPS_DOUBLE));
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
      return build_typecast(nearbyint_expr, int_type);
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
        index_result = build_typecast(index_result, double_type());
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
        const typet &symbol_type = sym->get_value().type().is_not_nil()
                                     ? sym->get_value().type()
                                     : sym->get_type();
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

          // Handle runtime conditionals that select between two string literals:
          // if cond then "a" else "b" -> if cond then complex(a) else complex(b).
          const exprt &sym_val = sym->get_value();
          if (sym_val.id() == "if" && sym_val.operands().size() == 3)
          {
            const exprt &cond = sym_val.operands()[0];

            symbolt true_sym;
            true_sym.set_value(sym_val.operands()[1]);
            true_sym.set_type(true_sym.get_value().type());
            auto true_text = extract_string_from_symbol(&true_sym);

            symbolt false_sym;
            false_sym.set_value(sym_val.operands()[2]);
            false_sym.set_type(false_sym.get_value().type());
            auto false_text = extract_string_from_symbol(&false_sym);

            auto parse_complex_text =
              [&](const std::optional<std::string> &text)
              -> std::optional<std::pair<double, double>> {
              if (!text)
                return std::nullopt;
              double real = 0.0, imag = 0.0;
              if (!complex_utils::parse_complex_string(*text, real, imag))
                return std::nullopt;
              return std::make_pair(real, imag);
            };

            auto true_complex = parse_complex_text(true_text);
            auto false_complex = parse_complex_text(false_text);

            if (cond.is_true())
            {
              if (!true_complex)
                return raise_value_error("complex() arg is a malformed string");
              return make_complex(
                from_double(true_complex->first, double_type()),
                from_double(true_complex->second, double_type()));
            }

            if (cond.is_false())
            {
              if (!false_complex)
                return raise_value_error("complex() arg is a malformed string");
              return make_complex(
                from_double(false_complex->first, double_type()),
                from_double(false_complex->second, double_type()));
            }

            if (true_complex && false_complex)
            {
              // V.3: build the per-part conditional selects in IREP2 (both
              // branches are double constants, so the if2t types agree).
              const type2tc dbl2 = double_type2();
              expr2tc cond2, tr2, fr2, ti2, fi2;
              migrate_expr(cond, cond2);
              migrate_expr(
                from_double(true_complex->first, double_type()), tr2);
              migrate_expr(
                from_double(false_complex->first, double_type()), fr2);
              migrate_expr(
                from_double(true_complex->second, double_type()), ti2);
              migrate_expr(
                from_double(false_complex->second, double_type()), fi2);
              exprt real_part = migrate_expr_back(if2tc(dbl2, cond2, tr2, fr2));
              exprt imag_part = migrate_expr_back(if2tc(dbl2, cond2, ti2, fi2));
              return make_complex(real_part, imag_part);
            }

            if (!true_complex && !false_complex)
            {
              return raise_value_error("complex() arg is a malformed string");
            }
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
    // Detect textual string operands (array of char / pointer to char / a
    // bare char) whose compile-time value could not be extracted above. The
    // existing constant-folding paths handle literals and constant-folded
    // conditionals; anything else (function parameters, return values from
    // calls, etc.) reaches here with a string-typed value but no constant
    // contents. Typecasting such a value to double generates an ill-typed
    // SMT expression that aborts the encoder, so reject it explicitly.
    auto is_char_subtype = [](const typet &t) {
      return t == char_type() ||
             (t.is_signedbv() && to_signedbv_type(t).get_width() == 8) ||
             (t.is_unsignedbv() && to_unsignedbv_type(t).get_width() == 8);
    };
    const typet &vt = value.type();
    const bool is_textual_array =
      vt.is_array() && is_char_subtype(vt.subtype());
    const bool is_textual_pointer =
      vt.is_pointer() && is_char_subtype(vt.subtype());
    const bool is_bare_char =
      is_char_subtype(vt) && !vt.is_array() && !vt.is_pointer();

    if (vt.is_array() && !is_textual_array)
      return raise_type_error(
        "complex() first argument must be a string or a number, not 'bytes'");

    if (is_textual_array || is_textual_pointer || is_bare_char)
      throw std::runtime_error(
        "complex() does not support non-literal string arguments");

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
      value = build_typecast(value, double_type());
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

  const bool real_is_complex = is_complex_type(real_arg.type());
  const bool imag_is_complex = is_complex_type(imag_arg.type());

  // Fast path for complex(real, imag) where both are plain real numerics.
  // Building x + y*1j through complex arithmetic can lose the sign bit of
  // signed zero in the imaginary part; preserve it by constructing the
  // complex value directly.
  if (!real_is_complex && !imag_is_complex)
  {
    if (real_arg.type() != double_type())
      real_arg = build_typecast(real_arg, double_type());
    if (imag_arg.type() != double_type())
      imag_arg = build_typecast(imag_arg, double_type());
    return make_complex(real_arg, imag_arg);
  }

  // Python semantics: complex(x, y) == x + y * 1j, including complex args.
  real_arg = promote_to_complex(real_arg);
  imag_arg = promote_to_complex(imag_arg);

  exprt a = build_member(real_arg, "real", double_type());
  exprt b = build_member(real_arg, "imag", double_type());
  exprt c = build_member(imag_arg, "real", double_type());
  exprt d = build_member(imag_arg, "imag", double_type());

  exprt real_part("ieee_sub", double_type());
  real_part.copy_to_operands(a, d);

  exprt imag_part("ieee_add", double_type());
  imag_part.copy_to_operands(b, c);

  return make_complex(real_part, imag_part);
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
          build_member(arg, components[0].get_name(), components[0].type());

        // Compare with remaining elements
        for (size_t i = 1; i < components.size(); ++i)
        {
          exprt elem =
            build_member(arg, components[i].get_name(), components[i].type());

          // result = (elem < result) ? elem : result  (> for max).
          // V.3: build the select in IREP2 when both branches share the
          // result type; mixed-type tuple components keep the legacy builder
          // (if2t asserts type-id equality).
          const typet &ut = components[i].type();
          if (elem.type() == ut && result.type() == ut)
          {
            const type2tc ut2 = migrate_type(ut);
            expr2tc elem2, result2;
            migrate_expr(elem, elem2);
            migrate_expr(result, result2);
            expr2tc cond = comparison_op == exprt::i_lt
                             ? lessthan2tc(elem2, result2)
                             : greaterthan2tc(elem2, result2);
            result = migrate_expr_back(if2tc(ut2, cond, elem2, result2));
          }
          else
          {
            exprt condition(comparison_op, type_handler_.get_typet("bool", 0));
            condition.copy_to_operands(elem, result);
            if_exprt update(condition, elem, result);
            update.type() = ut;
            result = update;
          }
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
      e = build_typecast(e, result_type);

  // Fold: result = exprs[0]; for each subsequent arg update via if-expr.
  // V.3: build the min/max selection chain in IREP2. All args are already
  // promoted to result_type, so the if2t branch types agree.
  const type2tc rt2 = migrate_type(result_type);
  expr2tc result2;
  migrate_expr(exprs[0], result2);
  for (size_t i = 1; i < exprs.size(); ++i)
  {
    expr2tc e2;
    migrate_expr(exprs[i], e2);
    expr2tc cond = comparison_op == exprt::i_lt ? lessthan2tc(e2, result2)
                                                : greaterthan2tc(e2, result2);
    result2 = if2tc(rt2, cond, e2, result2);
  }

  return migrate_expr_back(result2);
}

exprt function_call_expr::handle_print() const
{
  // Materialize each argument as code so arithmetic checks and function-call
  // side effects are preserved even though print itself has no runtime output.
  const auto &args = call_["args"];
  for (const auto &arg_node : args)
  {
    exprt arg_expr = converter_.get_expr(arg_node);

    // get_expr() on a Call node returns a code_function_callt (statement-form
    // expression), not the call's return value. Emit it directly so the
    // callee's side effects reach the GOTO program; the print return value
    // would otherwise be discarded along with the call itself.
    if (arg_expr.is_code() && arg_expr.get("statement") == "function_call")
    {
      converter_.current_block->copy_to_operands(to_code(arg_expr));
      continue;
    }

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

exprt function_call_expr::compute_element_truthiness(const exprt &element) const
{
  // V.3: build the scalar truthiness predicate in IREP2.
  if (element.type() == none_type())
    return migrate_expr_back(gen_false_expr());

  if (element.type().is_bool())
    return element;

  if (
    element.type().id() == "signedbv" || element.type().id() == "unsignedbv" ||
    element.type().id() == "floatbv" || element.type().is_pointer())
  {
    // element != 0
    expr2tc el2;
    migrate_expr(element, el2);
    return migrate_expr_back(not2tc(equality2tc(el2, gen_zero(el2->type))));
  }

  if (is_complex_type(element.type()))
    return complex_to_bool_expr(element);

  // For other types, assume truthy (conservative)
  return migrate_expr_back(gen_true_expr());
}

exprt function_call_expr::handle_any_all(ReduceOp op, const char *name)
{
  const std::string prefix = std::string(name) + "()";

  const auto keywords =
    call_.contains("keywords") ? call_["keywords"] : nlohmann::json::array();
  if (!keywords.empty())
    throw std::runtime_error(prefix + " takes no keyword arguments");

  const auto &args = call_["args"];

  if (args.empty())
    throw std::runtime_error(prefix + " expected at least 1 argument, got 0");

  if (args.size() > 1)
    throw std::runtime_error(
      prefix + " takes at most 1 argument, got " + std::to_string(args.size()));

  const auto &arg = args[0];
  const std::string &arg_type = arg["_type"];

  if (arg_type == "List" || arg_type == "Tuple" || arg_type == "Set")
    return reduce_iterable_literal_truthiness(arg, op);

  // Non-literal argument: forward to the Python operational model only
  // when the value is actually a list/set (pointer to PyListObj). Tuple
  // values are evaluated by combining the truthiness of each struct
  // member; anything else gets a clear error rather than being silently
  // passed to __ESBMC_list_size, which would dereference a non-list
  // pointer (issue #4295).
  exprt arg_expr = converter_.get_expr(arg);
  if (converter_.get_tuple_handler().is_tuple_type(arg_expr.type()))
    return reduce_tuple_expr_truthiness(arg_expr, op);

  // Sets share the PyListObject* representation, so the list-backed model
  // handles set variables transparently.
  if (arg_expr.type() == converter_.get_type_handler().get_list_type())
    return handle_general_function_call();

  throw std::runtime_error(
    prefix +
    " currently only supports list/tuple/set literals or list, set, or "
    "tuple variables");
}

exprt function_call_expr::handle_any()
{
  return handle_any_all(ReduceOp::Any, "any");
}

exprt function_call_expr::handle_all()
{
  return handle_any_all(ReduceOp::All, "all");
}

exprt function_call_expr::reduce_iterable_literal_truthiness(
  const nlohmann::json &iterable_arg,
  ReduceOp op) const
{
  const auto &elts = iterable_arg["elts"];

  if (elts.empty())
    // V.3: empty reduction -> all()=True, any()=False (built in IREP2).
    return migrate_expr_back(
      op == ReduceOp::All ? gen_true_expr() : gen_false_expr());

  std::optional<exprt> result;
  for (const auto &elt : elts)
  {
    exprt is_truthy = is_empty_literal(elt)
                        ? migrate_expr_back(gen_false_expr()) // V.3
                        : compute_element_truthiness(converter_.get_expr(elt));
    result = result ? combine_truthiness(std::move(*result), is_truthy, op)
                    : is_truthy;
  }
  return *result;
}

exprt function_call_expr::reduce_tuple_expr_truthiness(
  const exprt &tuple_expr,
  ReduceOp op) const
{
  const struct_typet &tuple_type = to_struct_type(tuple_expr.type());
  const auto &components = tuple_type.components();

  if (components.empty())
    // V.3: empty reduction -> all()=True, any()=False (built in IREP2).
    return migrate_expr_back(
      op == ReduceOp::All ? gen_true_expr() : gen_false_expr());

  std::optional<exprt> result;
  for (const auto &component : components)
  {
    exprt member =
      build_member(tuple_expr, component.get_name(), component.type());
    exprt is_truthy = compute_element_truthiness(member);
    result = result ? combine_truthiness(std::move(*result), is_truthy, op)
                    : is_truthy;
  }
  return *result;
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
  call.function() = build_symbol(*comb_func);
  call.type() = int_type();
  call.arguments().push_back(n_expr);
  call.arguments().push_back(k_expr);

  return call;
}
