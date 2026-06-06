#include <python-frontend/converter/converter_internal.h>
#include <python-frontend/function_call/expr.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_dict_handler.h>
#include <python-frontend/python_list.h>
#include <python-frontend/python_math.h>
#include <python-frontend/string/string_handler.h>
#include <python-frontend/tuple_handler.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <irep2/irep2_utils.h>
#include <util/arith_tools.h>
#include <util/c_typecast.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/migrate.h>
#include <util/python_types.h>
#include <util/std_code.h>
#include <util/std_expr.h>
#include <algorithm>
#include <cctype>
#include <sstream>

exprt python_converter::get_logical_operator_expr(const nlohmann::json &element)
{
  // `and`/`or` short-circuit: a later operand may not execute. get_named_expr
  // emits a walrus binding unconditionally, so a walrus inside any operand
  // would bind even when Python would skip it. Refuse with a clean diagnostic
  // rather than return an unsound verdict.
  if (element.contains("values") && contains_named_expr(element["values"]))
    throw std::runtime_error(
      "Walrus operator ':=' in a boolean (and/or) operand is not supported");

  std::string op(element["op"]["_type"].get<std::string>());
  exprt logical_expr(
    python_frontend::map_operator(op, bool_type()), bool_type());
  bool contains_non_boolean = false;
  auto get_truthy_condition = [&](const exprt &value_expr) -> exprt {
    typet list_type = type_handler_.get_list_type();
    if (
      value_expr.type() == list_type ||
      (value_expr.type().is_pointer() &&
       value_expr.type().subtype() == list_type))
    {
      const symbolt *size_func =
        symbol_table_.find_symbol("c:@F@__ESBMC_list_size");
      if (!size_func)
        throw std::runtime_error(
          "__ESBMC_list_size not found for list condition check");

      side_effect_expr_function_callt size_call;
      size_call.function() = symbol_expr(*size_func);
      size_call.type() = size_type();
      size_call.location() = get_location_from_decl(element);
      if (value_expr.type().is_pointer())
        size_call.arguments().push_back(value_expr);
      else
        size_call.arguments().push_back(address_of_exprt(value_expr));

      exprt cond("notequal", bool_type());
      cond.copy_to_operands(size_call, gen_zero(size_type()));
      cond.location() = get_location_from_decl(element);
      return cond;
    }

    return value_expr;
  };

  // Mark that we're processing operands in an expression context
  // This ensures boolean-returning function calls are converted to side-effect expressions
  bool old_is_converting_rhs = is_converting_rhs;
  is_converting_rhs = true;

  // Iterate over operands of logical operations (and/or)
  for (const auto &operand : element["values"])
  {
    exprt operand_expr = get_expr(operand);
    if (operand_expr.is_code() && operand_expr.statement() == "function_call")
    {
      const code_function_callt &code_call =
        to_code_function_call(to_code(operand_expr));
      typet return_type = code_call.type();
      if (return_type.is_empty() || return_type.id() == typet::t_empty)
        return_type = type_handler_.get_typet("int", 0);
      side_effect_expr_function_callt side_effect_call;
      side_effect_call.function() = code_call.function();
      side_effect_call.arguments() = code_call.arguments();
      side_effect_call.type() = return_type;
      side_effect_call.location() = code_call.location();
      operand_expr = side_effect_call;
    }
    logical_expr.copy_to_operands(operand_expr);
    contains_non_boolean |= !operand_expr.is_boolean();
  }

  // Restore the original flag state
  is_converting_rhs = old_is_converting_rhs;

  // A BoolOp must have at least two values, but AST rewrites (e.g. lowering
  // `x == []` to `len(x) == 0`) can produce a degenerate one-value node. A
  // single-operand `and`/`or` is malformed IR: the C adjuster reads op1() and
  // runs off the end of the operands vector. Collapse to the lone operand,
  // which is the correct result for a one-value boolean operation.
  if (logical_expr.operands().size() == 1)
    return logical_expr.operands().front();

  // Shockingly enough, a BoolOp may not return a boolean.
  if (contains_non_boolean)
  {
    typet t = extract_type_from_boolean_op(logical_expr).type();
    // Are we dealing with an actual bool expression?
    if (t.is_bool())
      return logical_expr;
    // Result expression starts from last operand as default else branch
    exprt result_expr = logical_expr.operands().back();
    for (int i = logical_expr.operands().size() - 2; i >= 0; i--)
    {
      const exprt &current = logical_expr.operands()[i];
      exprt current_cond = get_truthy_condition(current);
      exprt if_expr("if", t);
      if (logical_expr.is_and())
        if_expr.copy_to_operands(current_cond, result_expr, current);
      else
        if_expr.copy_to_operands(current_cond, current, result_expr);

      result_expr = if_expr;
    }
    return result_expr;
  }
  return logical_expr;
}
inline bool is_ieee_op(const exprt &expr)
{
  const std::string &id = expr.id().as_string();
  return id == "ieee_add" || id == "ieee_mul" || id == "ieee_sub" ||
         id == "ieee_div";
}

// Attach source location from symbol table if expr is a symbol
static void attach_symbol_location(exprt &expr, contextt &symbol_table)
{
  if (!expr.is_symbol())
    return;

  const irep_idt &id = expr.identifier();
  symbolt *sym = symbol_table.find_symbol(id);
  if (sym != nullptr)
    expr.location() = sym->location;
}

std::string python_converter::get_python_type_category(const typet &t) const
{
  // Unannotated any_type (void*) — caller keeps the existing coercion path.
  if (t.is_pointer() && t.subtype().id() == "empty")
    return "";

  // Python has no `char`: single-character results from `chr()` and string
  // indexing are 1-char strings. ESBMC models them as 8-bit integers tagged
  // with `#cpp_type==char` (set explicitly in type_handler::get_typet for
  // `chr()` and in python_list::index for string subscript). A bare width-8
  // int *without* the marker is something else (e.g. `dtype=np.int8`,
  // 8-bit user int annotation) and must remain in the numeric tower —
  // misclassifying it as "string" turns `np.add(127, 1, dtype=np.int8) ==
  // -128` into a spurious cross-type fold to False.
  if (type_utils::is_string_type(t) || type_utils::is_char_type(t))
    return "string";

  // Python's numeric tower coerces within itself; complex is checked first
  // because it is a struct and would otherwise fall through to class_inst.
  if (
    is_complex_type(t) || t.is_bool() || t.is_floatbv() ||
    type_utils::is_integer_type(t))
    return "numeric";

  // Any remaining non-string array (e.g. bytes literal `b'A'`).
  if (t.is_array())
    return "bytes";

  if (t == type_handler_.get_list_type())
    return "list";

  if (dict_handler_->is_dict_type(t))
    return "dict";

  if (tuple_handler_->is_tuple_type(t))
    return "tuple";

  // User-defined class instances are deliberately reported as unknown so the
  // caller falls through to `dispatch_dunder_operator`, which honours a
  // user-defined `__eq__`. A cross-type fold here would silently bypass it.
  return "";
}

exprt handle_float_vs_string(exprt &bin_expr, const std::string &op)
{
  if (op == "Eq")
  {
    // float == str → False (no exception)
    bin_expr.make_false();
  }
  else if (op == "NotEq")
  {
    // float != str → True (no exception)
    bin_expr.make_true();
  }
  else if (type_utils::is_ordered_comparison(op))
  {
    // Python-style error: float < str → TypeError
    std::string lower_op = op;
    std::transform(
      lower_op.begin(), lower_op.end(), lower_op.begin(), [](unsigned char c) {
        return std::tolower(c);
      });

    const auto &loc = bin_expr.location();
    const auto &op_map = python_frontend::operator_map();
    const auto it = op_map.find(lower_op);
    assert(it != op_map.end());

    std::ostringstream error;
    error << "'" << it->second
          << "' not supported between instances of 'float' and 'str'";

    if (loc.is_not_nil())
      error << " at " << loc.get_file() << ":" << loc.get_line();
    else
      error << " at <unknown location>";

    throw std::runtime_error(error.str());
  }

  return bin_expr;
}
void python_converter::convert_function_calls_to_side_effects(
  exprt &lhs,
  exprt &rhs)
{
  auto to_side_effect_call = [](exprt &expr) {
    side_effect_expr_function_callt side_effect;
    code_function_callt &code = static_cast<code_function_callt &>(expr);
    side_effect.function() = code.function();
    side_effect.location() = code.location();
    side_effect.type() = code.type();
    side_effect.arguments() = code.arguments();
    expr = side_effect;
  };

  if (lhs.is_function_call())
    to_side_effect_call(lhs);
  if (rhs.is_function_call())
    to_side_effect_call(rhs);
}

/// Handle chained comparisons
exprt python_converter::handle_chained_comparisons_logic(
  const nlohmann::json &element,
  exprt &bin_expr)
{
  exprt cond("and", bool_type());
  cond.move_to_operands(bin_expr); // bin_expr compares left and comparators[0]

  for (size_t i = 0; i + 1 < element["comparators"].size(); ++i)
  {
    std::string op(element["ops"][i + 1]["_type"].get<std::string>());
    exprt logical_expr(
      python_frontend::map_operator(op, bool_type()), bool_type());
    exprt op1 = get_expr(element["comparators"][i]);
    exprt op2 = get_expr(element["comparators"][i + 1]);

    convert_function_calls_to_side_effects(op1, op2);

    std::string op1_type = type_handler_.type_to_string(op1.type());
    std::string op2_type = type_handler_.type_to_string(op2.type());

    if (op1_type == "str" && op2_type == "str")
    {
      exprt string_expr = handle_string_comparison(op, op1, op2, element);
      if (string_expr.is_nil())
      {
        exprt expr(python_frontend::map_operator(op, bool_type()), bool_type());
        expr.copy_to_operands(op1, op2);
        cond.move_to_operands(expr);
      }
      else
      {
        cond.move_to_operands(string_expr);
      }
    }
    else
    {
      logical_expr.copy_to_operands(op1);
      logical_expr.copy_to_operands(op2);
      cond.move_to_operands(logical_expr);
    }
  }
  return cond;
}

exprt python_converter::handle_membership_operator(
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &element,
  bool invert)
{
  // Check if rhs is a dictionary (struct type with dict tag)
  typet rhs_resolved_type = rhs.type();
  if (rhs.is_symbol())
  {
    const symbolt *sym = symbol_table_.find_symbol(rhs.identifier());
    if (sym)
      rhs_resolved_type = sym->get_type();
  }

  if (rhs_resolved_type.id() == "symbol")
    rhs_resolved_type = ns.follow(rhs_resolved_type);

  if (rhs_resolved_type.is_struct())
  {
    const struct_typet &struct_type = to_struct_type(rhs_resolved_type);
    std::string tag = struct_type.tag().as_string();

    if (
      tag.find("dict_") != std::string::npos ||
      tag.find("tag-dict") != std::string::npos)
    {
      return dict_handler_->handle_dict_membership(lhs, rhs, invert);
    }

    if (tag.starts_with("tag-tuple"))
    {
      // `x in (a, b, c)` is element-wise equality, with string elements
      // compared by content. handle_tuple_membership builds the OR chain.
      return tuple_handler_->handle_tuple_membership(lhs, rhs, invert, element);
    }
  }

  typet list_type = type_handler_.get_list_type();

  // Handle set/list membership:
  // "item" in [list/set] or "item" not in [list/set]
  if (rhs.type() == list_type)
  {
    python_list list(*this, element);
    exprt contains_expr = list.contains(lhs, rhs);
    return invert ? not_exprt(contains_expr) : contains_expr;
  }

  // Get string type identifiers
  std::string lhs_type = type_handler_.type_to_string(lhs.type());
  std::string rhs_type = type_handler_.type_to_string(rhs.type());

  // Handle string membership testing: "substr" in "string" or "substr" not in "string"
  if (
    lhs.type().is_pointer() || rhs.type().is_pointer() ||
    lhs.type().is_array() || rhs.type().is_array() || lhs_type == "str" ||
    rhs_type == "str")
  {
    exprt membership_expr =
      string_handler_.handle_string_membership(lhs, rhs, element);
    return invert ? not_exprt(membership_expr) : membership_expr;
  }

  throw std::runtime_error(
    std::string("Unsupported expression for '") + (invert ? "not in" : "in") +
    "' operation");
}

exprt python_converter::get_binary_operator_expr(const nlohmann::json &element)
{
  // Extract left and right operands from AST
  auto left = element.contains("left") ? element["left"] : element["target"];

  decltype(left) right;
  if (element.contains("right"))
    right = element["right"];
  else if (element.contains("comparators"))
    right = element["comparators"][0];
  else if (element.contains("value"))
    right = element["value"];

  // Convert operands to expressions
  exprt lhs = get_expr(left);
  exprt rhs = get_expr(right);

  // Resolve dictionary subscript types for proper comparison
  dict_handler_->resolve_dict_subscript_types(left, right, lhs, rhs);

  // Extract operator
  std::string op;
  if (element.contains("op"))
    op = element["op"]["_type"].get<std::string>();
  else if (element.contains("ops"))
    op = element["ops"][0]["_type"].get<std::string>();
  assert(!op.empty());

  // Handle type identity checks (e.g., y is int, x is str)
  exprt type_identity_result =
    handle_type_identity_check(op, lhs, rhs, left, right);
  if (!type_identity_result.is_nil())
    return type_identity_result;

  // Handle None comparisons (don't unwrap optionals for identity checks)
  bool is_none_check = handle_none_check_setup(op, lhs, rhs);
  if (!is_none_check)
  {
    lhs = unwrap_optional_if_needed(lhs, element);
    rhs = unwrap_optional_if_needed(rhs, element);
  }

  if (lhs.type() == none_type() || rhs.type() == none_type())
  {
    // A direct call like `f() is None` arrives as code_function_callt — a
    // statement, not a value. Promote to side_effect_expr_function_callt so
    // it carries the return type and downstream isnone simplification can
    // type-check it as a pointer/struct rather than as empty code.
    lhs = to_value_expr(lhs, ns);
    rhs = to_value_expr(rhs, ns);
    return handle_none_comparison(op, lhs, rhs);
  }

  // Handle exceptions
  if (lhs.statement() == "cpp-throw")
    return lhs;
  if (rhs.statement() == "cpp-throw")
    return rhs;

  attach_symbol_location(lhs, symbol_table());
  attach_symbol_location(rhs, symbol_table());

  // Python rule: cross-type `==`/`!=` returns False/True without coercion,
  // except within the numeric tower (bool/int/float/complex). Fold here, before
  // the dict/list/tuple/SMT-encoding paths that otherwise crash on mismatched
  // operands (e.g. float vs bytes, dict vs int, list vs int — issue #4628).
  if (op == "Eq" || op == "NotEq")
  {
    const std::string lc = get_python_type_category(lhs.type());
    const std::string rc = get_python_type_category(rhs.type());
    if (!lc.empty() && !rc.empty() && lc != rc)
      return gen_boolean(op == "NotEq");
  }

  // Handle set operations (difference, intersection, union)
  typet list_type = type_handler_.get_list_type();
  if (
    (lhs.type() == list_type || rhs.type() == list_type) &&
    (op == "Sub" || op == "BitAnd" || op == "BitOr"))
  {
    exprt set_result =
      python_set::handle_operations(*this, op, lhs, rhs, element);
    if (!set_result.is_nil())
      return set_result;
  }

  // Handle membership operators
  if (op == "In")
    return handle_membership_operator(lhs, rhs, element, false);
  if (op == "NotIn")
    return handle_membership_operator(lhs, rhs, element, true);

  // Convert function calls to side effects
  convert_function_calls_to_side_effects(lhs, rhs);
  if (lhs.statement() == "cpp-throw")
    return lhs;
  if (rhs.statement() == "cpp-throw")
    return rhs;

  // Dispatch dunder methods for user-defined struct types
  {
    exprt dunder_result =
      dispatch_dunder_operator(op, lhs, rhs, get_location_from_decl(element));
    if (!dunder_result.is_nil())
      return dunder_result;
  }

  // Intercept complex arithmetic/comparison before generic binary expression
  // building, which cannot operate directly on struct operands.
  if (is_complex_type(lhs.type()) || is_complex_type(rhs.type()))
    return complex_handler_.handle_binary_op(op, lhs, rhs, element);

  // Handle array/string operations
  if (lhs.type().is_array() || rhs.type().is_array())
  {
    exprt result = handle_array_operations(op, lhs, rhs, left, right, element);
    if (!result.is_nil())
      return result;
  }

  // Handle dictionary comparison
  if (
    lhs.type().is_struct() && rhs.type().is_struct() &&
    dict_handler_->is_dict_type(lhs.type()) &&
    dict_handler_->is_dict_type(rhs.type()) && (op == "Eq" || op == "NotEq"))
  {
    // Fold literal-vs-literal dict equality at conversion time. Skipping the
    // O(n^2) __ESBMC_dict_eq runtime model here is the dominant win under
    // --incremental-bmc, where each k iteration would otherwise re-symbolise
    // the comparison from scratch (issue #4623).
    if (auto folded = dict_handler_->try_constant_fold_eq(left, right))
      return gen_boolean(op == "NotEq" ? !*folded : *folded);
    return dict_handler_->compare(lhs, rhs, op);
  }

  // Handle tuple +/* before list operations (tuple structs would otherwise
  // fall through to the generic struct-arithmetic path and silently produce
  // a struct of the original size).
  exprt tuple_result = handle_tuple_operations(op, lhs, rhs, element);
  if (!tuple_result.is_nil())
    return tuple_result;

  // Handle list operations
  exprt list_result =
    handle_list_operations(op, lhs, rhs, left, right, element);
  if (!list_result.is_nil())
    return list_result;

  // Python reference-identity for class instances: class objects are stored
  // by value (tag-X) while attribute chains reach them through pointer fields
  // (tag-X *). When one operand is a struct instance and the other is a
  // pointer to the same class, take the address of the struct so both sides
  // compare as references. Covers `a.next.next == a` / `is a` on cyclic
  // linked structures (github #4116).
  if (
    (op == "Eq" || op == "NotEq" || op == "Is" || op == "IsNot") &&
    lhs.type().is_pointer() != rhs.type().is_pointer())
  {
    exprt &struct_side = lhs.type().is_pointer() ? rhs : lhs;
    const exprt &ptr_side = lhs.type().is_pointer() ? lhs : rhs;
    auto class_tag = [&](const typet &t) -> irep_idt {
      typet r = t;
      if (r.id() == "symbol")
        r = ns.follow(r);
      return r.is_struct() ? to_struct_type(r).tag() : irep_idt();
    };
    const irep_idt s_tag = class_tag(struct_side.type());
    if (!s_tag.empty() && s_tag == class_tag(ptr_side.type().subtype()))
      struct_side = gen_address_of(struct_side);
  }

  // Handle identity comparisons
  if (op == "Is")
    return get_binary_operator_expr_for_is(lhs, rhs);
  if (op == "IsNot")
    return get_negated_is_expr(lhs, rhs);

  // Handle relational operation type mismatches
  if (type_utils::is_relational_op(op))
  {
    exprt result = handle_relational_type_mismatches(op, lhs, rhs, element);
    if (!result.is_nil())
      return result;
  }

  // Handle string operations
  exprt string_result =
    handle_string_binary_operations(op, lhs, rhs, left, right, element);
  if (!string_result.is_nil())
  {
    if (element.contains("comparators") && element["comparators"].size() > 1)
      return handle_chained_comparisons_logic(element, string_result);
    return string_result;
  }

  // void* vs string: emit `strcmp(a, b) op 0`
  // instead of decaying to pointer equality or returning a static False.
  // Hits when one side is an unannotated class attribute holding a string.
  if (op == "Eq" || op == "NotEq")
  {
    auto is_void_ptr = [](const typet &t) {
      return t.is_pointer() && t.subtype().id() == "empty";
    };
    auto is_string_like = [](const typet &t) {
      return (t.is_array() || t.is_pointer()) && t.subtype() == char_type();
    };
    const bool lhs_void = is_void_ptr(lhs.type());
    const bool rhs_void = is_void_ptr(rhs.type());
    if (
      (lhs_void && is_string_like(rhs.type())) ||
      (rhs_void && is_string_like(lhs.type())))
    {
      const symbolt *strcmp_symbol = symbol_table_.find_symbol("c:@F@strcmp");
      if (strcmp_symbol)
      {
        // Normalise each side to char*: cast void*, take base address of arrays.
        const typet char_ptr = pointer_typet(char_type());
        auto as_char_ptr = [&](const exprt &e) -> exprt {
          if (is_void_ptr(e.type()))
            return typecast_exprt(e, char_ptr);
          if (e.type().is_array())
            return string_handler_.get_array_base_address(e);
          return e;
        };

        side_effect_expr_function_callt strcmp_call;
        strcmp_call.function() = symbol_expr(*strcmp_symbol);
        strcmp_call.arguments() = {as_char_ptr(lhs), as_char_ptr(rhs)};
        strcmp_call.location() = get_location_from_decl(element);
        strcmp_call.type() = int_type();

        exprt result(
          python_frontend::map_operator(op, bool_type()), bool_type());
        result.copy_to_operands(strcmp_call, gen_zero(int_type()));
        result.location() = get_location_from_decl(element);
        return result;
      }
    }
  }

  // Handle type mismatches
  exprt type_mismatch_result = handle_string_type_mismatch(lhs, rhs, op);
  if (!type_mismatch_result.is_nil())
    return type_mismatch_result;

  // Detect any_type (void*) operands — unannotated Python parameters.
  auto is_any_ptr = [](const exprt &e) {
    return e.type().is_pointer() && e.type().subtype().id() == "empty";
  };
  auto is_integer = [](const exprt &e) {
    return e.type().is_signedbv() || e.type().is_unsignedbv();
  };

  // For arithmetic operations (Sub, Add, Mult, etc.) on an any_type (void*)
  // operand combined with an integer operand, cast the void* to the integer type.
  if (
    !type_utils::is_relational_op(op) && op != "Is" && op != "IsNot" &&
    op != "In" && op != "NotIn")
  {
    if (is_any_ptr(lhs) && is_any_ptr(rhs))
    {
      const typet int_type = type_handler_.get_typet("int", 0);
      lhs = typecast_exprt(lhs, int_type);
      rhs = typecast_exprt(rhs, int_type);
    }
    else if (is_any_ptr(lhs) && is_integer(rhs))
      lhs = typecast_exprt(lhs, rhs.type());
    else if (is_any_ptr(rhs) && is_integer(lhs))
      rhs = typecast_exprt(rhs, lhs.type());
  }

  // Handle Any-typed (void*) operands in comparisons.
  if (
    type_utils::is_ordered_comparison(op) || op == "Eq" || op == "NotEq" ||
    op == "Is" || op == "IsNot" || op == "In" || op == "NotIn")
  {
    auto cast_to_void_ptr = [](exprt &e, const typet &ptr_type) {
      if (e.type().is_floatbv())
      {
        unsigned width = static_cast<const bv_typet &>(e.type()).get_width();
        exprt bitcast("bitcast", unsignedbv_typet(width));
        bitcast.copy_to_operands(e);
        e = bitcast;
      }
      e = typecast_exprt(e, ptr_type);
    };
    if (is_any_ptr(lhs) && !is_any_ptr(rhs) && !rhs.type().is_pointer())
    {
      if (type_utils::is_ordered_comparison(op) && is_integer(rhs))
        lhs = typecast_exprt(lhs, rhs.type()); // cast void* to integer
      else
        cast_to_void_ptr(rhs, lhs.type()); // cast integer to void*
    }
    else if (is_any_ptr(rhs) && !is_any_ptr(lhs) && !lhs.type().is_pointer())
    {
      if (type_utils::is_ordered_comparison(op) && is_integer(lhs))
        rhs = typecast_exprt(rhs, lhs.type()); // cast void* to integer
      else
        cast_to_void_ptr(lhs, rhs.type()); // cast integer to void*
    }

    // Optional[T] is lowered to T* (see converter_types.cpp:457) and a
    // non-None T value is round-tripped through (T*) at the call site
    // (e.g. foo(5) -> y = (int*)5). Make equality with a matching primitive
    // work by casting the primitive side to the pointer type, so the
    // resulting comparison matches the round-tripped address. Skip ordered
    // comparisons: they would silently compare addresses, not values.
    auto is_primitive_ptr = [](const exprt &e) {
      if (!e.type().is_pointer())
        return false;
      const typet &sub = e.type().subtype();
      return sub.is_signedbv() || sub.is_unsignedbv() || sub.is_floatbv();
    };
    auto is_primitive = [](const exprt &e) {
      return e.type().is_signedbv() || e.type().is_unsignedbv() ||
             e.type().is_floatbv();
    };
    if (op == "Eq" || op == "NotEq" || op == "Is" || op == "IsNot")
    {
      if (
        is_primitive_ptr(lhs) && is_primitive(rhs) &&
        lhs.type().subtype() == rhs.type())
        cast_to_void_ptr(rhs, lhs.type());
      else if (
        is_primitive_ptr(rhs) && is_primitive(lhs) &&
        rhs.type().subtype() == lhs.type())
        cast_to_void_ptr(lhs, rhs.type());
    }
  }

  // Handle special mathematical operations
  if (op == "Pow" || op == "power")
    return math_handler_.handle_power(lhs, rhs);

  if (op == "Mod" && (lhs.type().is_floatbv() || rhs.type().is_floatbv()))
    return math_handler_.handle_modulo(lhs, rhs, element);

  if (type_utils::is_relational_op(op))
  {
    const bool lhs_invalid = lhs.type().is_empty() || lhs.type().is_nil();
    const bool rhs_invalid = rhs.type().is_empty() || rhs.type().is_nil();
    locationt loc = get_location_from_decl(element);

    // Sound over-approximation when the comparison cannot be lowered to a
    // typed binop: either an operand's type is unresolvable, or one side is
    // a pointer-backed value (e.g. a list/dict variable that has been
    // reassigned across incompatible types in the same scope) and the
    // other isn't. Aborting here loses an entire verification run for what
    // is often a frontend type-inference gap, not a real soundness issue;
    // returning nondet bool lets symbolic execution explore both outcomes
    // and keeps safety verification sound (we cannot conclude SAFE when
    // the real comparison would fail). See #4807.
    auto nondet_comparison = [&](const char *reason) {
      log_debug(
        "python-binop",
        "{} at {}:{} -- falling back to nondet bool",
        reason,
        loc.is_nil() ? std::string("<unknown>") : loc.get_file().as_string(),
        loc.is_nil() ? std::string("?") : loc.get_line().as_string());
      side_effect_expr_nondett nondet(bool_type());
      nondet.location() = loc;
      return nondet;
    };

    if (lhs_invalid || rhs_invalid)
      return nondet_comparison(
        "unsupported comparison with unresolved operand type");

    const bool lhs_ptr = lhs.type().is_pointer();
    const bool rhs_ptr = rhs.type().is_pointer();
    if (lhs_ptr != rhs_ptr)
      return nondet_comparison(
        "unsupported comparison between pointer-backed and non-pointer "
        "values");
  }

  // Build the binary expression
  exprt bin_expr = build_binary_expression(op, lhs, rhs);

  // Handle float vs char comparisons
  if (type_utils::is_float_vs_char(lhs, rhs))
    return handle_float_vs_string(bin_expr, op);

  // Handle floor division
  if (op == "FloorDiv")
    return math_handler_.handle_floor_division(lhs, rhs, bin_expr);

  // Python integer modulo `%` is floored (result takes the sign of the
  // divisor), unlike C's truncated remainder. Correct it for integer operands;
  // float `%` was already handled above via handle_modulo.
  if (
    op == "Mod" && (lhs.type().is_signedbv() || lhs.type().is_unsignedbv()) &&
    (rhs.type().is_signedbv() || rhs.type().is_unsignedbv()))
    return math_handler_.handle_int_modulo(lhs, rhs, bin_expr);

  // Promote operands for IEEE operations
  promote_ieee_operands(bin_expr, lhs, rhs);

  // Handle chained comparisons
  if (element.contains("comparators") && element["comparators"].size() > 1)
    return handle_chained_comparisons_logic(element, bin_expr);

  return bin_expr;
}

bool python_converter::handle_none_check_setup(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs)
{
  bool is_none_check = (op == "Is" || op == "IsNot") &&
                       (lhs.type() == none_type() || rhs.type() == none_type());

  if (!is_none_check && (op == "Eq" || op == "NotEq"))
  {
    if (lhs.type() == none_type() || rhs.type() == none_type())
      is_none_check = true;
  }

  return is_none_check;
}

exprt python_converter::handle_array_operations(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right,
  const nlohmann::json &element)
{
  if (!lhs.type().is_array() && !rhs.type().is_array())
  {
    if (
      op == "Mult" && lhs.type().is_pointer() && rhs.type().is_pointer() &&
      !type_utils::is_string_type(lhs.type()) &&
      !type_utils::is_string_type(rhs.type()))
    {
      std::ostringstream msg;
      msg << "TypeError: arithmetic on numeric arrays is not supported "
             "(numpy broadcasting is not modelled)";
      const locationt loc = get_location_from_decl(element);
      if (!loc.is_nil())
        msg << " at " << loc.get_file() << ":" << loc.get_line();
      throw std::runtime_error(msg.str());
    }
    return nil_exprt();
  }

  // Check for zero-length array comparisons
  if (
    string_handler_.is_zero_length_array(lhs) &&
    string_handler_.is_zero_length_array(rhs) && (op == "Eq" || op == "NotEq"))
  {
    return gen_boolean(op == "Eq");
  }

  auto is_numeric_type = [](const typet &t) {
    return t.is_signedbv() || t.is_unsignedbv() || t.is_floatbv();
  };

  auto is_char_array = [](const typet &t) {
    if (!t.is_array())
      return false;
    const typet &elt = t.subtype();
    return elt == char_type() ||
           (elt.is_signedbv() && to_signedbv_type(elt).get_width() == 8) ||
           (elt.is_unsignedbv() && to_unsignedbv_type(elt).get_width() == 8);
  };
  auto is_list_model_type = [this](const typet &t) {
    const typet list_type = type_handler_.get_list_type();
    if (t == list_type)
      return true;
    if (t.is_pointer() && ns.follow(t.subtype()) == list_type)
      return true;
    return false;
  };

  auto build_scalar_broadcast =
    [&](exprt &array_expr, exprt &scalar_expr) -> exprt {
    const typet &array_type = array_expr.type();
    if (!array_type.is_array())
      return nil_exprt();

    const typet &elem_type = array_type.subtype();
    if (!is_numeric_type(elem_type) || !is_numeric_type(scalar_expr.type()))
      return nil_exprt();

    const exprt &size_expr = to_array_type(array_type).size();
    if (!size_expr.is_constant())
      return nil_exprt();

    const BigInt size_big =
      binary2integer(to_constant_expr(size_expr).value().c_str(), true);
    if (size_big < 0)
      return nil_exprt();

    const long long array_size = size_big.to_int64();
    if (array_size < 0)
      return nil_exprt();

    exprt casted_scalar = scalar_expr.type() == elem_type
                            ? scalar_expr
                            : typecast_exprt(scalar_expr, elem_type);

    exprt result_array = array_expr;
    // V.3: build the array element access in IREP2 (index2tc), the exact
    // round-trip of index_exprt (behaviour-preserving). The array and element
    // type are loop-invariant, so migrate them once.
    expr2tc ar2;
    migrate_expr(array_expr, ar2);
    const type2tc elem_t2 = migrate_type(elem_type);
    for (long long i = 0; i < array_size; ++i)
    {
      exprt index = from_integer(i, size_type());
      expr2tc ix2;
      migrate_expr(index, ix2);
      exprt array_item = migrate_expr_back(index2tc(elem_t2, ar2, ix2));
      exprt bin_elem(python_frontend::map_operator(op, elem_type), elem_type);
      bin_elem.copy_to_operands(array_item, casted_scalar);
      result_array = with_exprt(result_array, index, bin_elem);
    }
    return result_array;
  };

  // For direct Python binary operators over arrays, keep behaviour explicit
  // and conservative: broadcasting is not modelled in this path.
  if ((op == "Add" || op == "Mult"))
  {
    const bool lhs_numeric_array = lhs.type().is_array() &&
                                   is_numeric_type(lhs.type().subtype()) &&
                                   !is_char_array(lhs.type());
    const bool rhs_numeric_array = rhs.type().is_array() &&
                                   is_numeric_type(rhs.type().subtype()) &&
                                   !is_char_array(rhs.type());

    // Numeric array +/-/* scalar (including chained forms) is supported here.
    if (lhs_numeric_array && !rhs.type().is_array())
    {
      exprt lowered = build_scalar_broadcast(lhs, rhs);
      if (!lowered.is_nil())
        return lowered;
    }
    // Keep scalar + array as unsupported for now (matches regression
    // expectations and avoids inconsistent with_expr construction paths).
    if (op == "Add" && rhs_numeric_array && !lhs.type().is_array())
    {
      std::ostringstream msg;
      msg << "TypeError: arithmetic on numeric arrays is not supported "
             "(numpy broadcasting is not modelled)";
      const locationt loc = get_location_from_decl(element);
      if (!loc.is_nil())
        msg << " at " << loc.get_file() << ":" << loc.get_line();
      throw std::runtime_error(msg.str());
    }
    if (op == "Mult" && rhs_numeric_array && !lhs.type().is_array())
    {
      std::ostringstream msg;
      msg << "TypeError: arithmetic on numeric arrays is not supported "
             "(numpy broadcasting is not modelled)";
      const locationt loc = get_location_from_decl(element);
      if (!loc.is_nil())
        msg << " at " << loc.get_file() << ":" << loc.get_line();
      throw std::runtime_error(msg.str());
    }

    // Keep unsupported combinations explicit and deterministic.
    if (op == "Mult" && lhs_numeric_array && rhs_numeric_array)
    {
      std::ostringstream msg;
      msg << "TypeError: arithmetic on numeric arrays is not supported "
             "(numpy broadcasting is not modelled)";
      const locationt loc = get_location_from_decl(element);
      if (!loc.is_nil())
        msg << " at " << loc.get_file() << ":" << loc.get_line();
      throw std::runtime_error(msg.str());
    }

    // Defensive guard for list-model arrays (e.g. 2D numpy lowering) to avoid
    // falling through to generic arithmetic paths that assert internally.
    if (
      op == "Mult" && is_list_model_type(lhs.type()) &&
      is_list_model_type(rhs.type()))
    {
      std::ostringstream msg;
      msg << "TypeError: arithmetic on numeric arrays is not supported "
             "(numpy broadcasting is not modelled)";
      const locationt loc = get_location_from_decl(element);
      if (!loc.is_nil())
        msg << " at " << loc.get_file() << ":" << loc.get_line();
      throw std::runtime_error(msg.str());
    }
  }

  // Handle string concatenation -- only valid for char-arrays. Numeric
  // arrays (e.g. ``np.array([1, 2, 3])``, modelled as ``long int [N]``)
  // reaching this path via ``a + n`` previously got force-cast to char*
  // and fed to __python_str_concat, which then crashed the bitwuzla
  // mk_store on the sort-width mismatch. Reject numeric-array arithmetic
  // explicitly: broadcasting is unsupported.
  if (op == "Add")
  {
    const bool lhs_char = is_char_array(lhs.type());
    const bool rhs_char = is_char_array(rhs.type());
    if (!lhs_char && !rhs_char)
    {
      std::ostringstream msg;
      msg << "TypeError: arithmetic on numeric arrays is not supported "
             "(numpy broadcasting is not modelled)";
      const locationt loc = get_location_from_decl(element);
      if (!loc.is_nil())
        msg << " at " << loc.get_file() << ":" << loc.get_line();
      throw std::runtime_error(msg.str());
    }
    return string_handler_.handle_string_concatenation_with_promotion(
      lhs, rhs, left, right);
  }

  return nil_exprt();
}

exprt python_converter::handle_tuple_operations(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &element)
{
  const bool lhs_is_tuple = tuple_handler_->is_tuple_type(lhs.type());
  const bool rhs_is_tuple = tuple_handler_->is_tuple_type(rhs.type());

  // Concatenation: (a, b) + (c, d) == (a, b, c, d).
  if (op == "Add" && lhs_is_tuple && rhs_is_tuple)
  {
    const auto &lhs_components = to_struct_type(lhs.type()).components();
    const auto &rhs_components = to_struct_type(rhs.type()).components();

    std::vector<typet> element_types;
    element_types.reserve(lhs_components.size() + rhs_components.size());
    for (const auto &c : lhs_components)
      element_types.push_back(c.type());
    for (const auto &c : rhs_components)
      element_types.push_back(c.type());

    struct_typet new_type =
      tuple_handler_->create_tuple_struct_type(element_types);

    struct_exprt result(new_type);
    // V.3: IREP2 tuple-component access (exact round-trip of member_exprt).
    expr2tc lhs2, rhs2;
    migrate_expr(lhs, lhs2);
    migrate_expr(rhs, rhs2);
    for (const auto &c : lhs_components)
      result.copy_to_operands(migrate_expr_back(
        member2tc(migrate_type(c.type()), lhs2, c.get_name())));
    for (const auto &c : rhs_components)
      result.copy_to_operands(migrate_expr_back(
        member2tc(migrate_type(c.type()), rhs2, c.get_name())));

    if (element.contains("lineno"))
      result.location() = get_location_from_decl(element);
    return result;
  }

  // Repetition: (a, b) * 3 == (a, b, a, b, a, b). Requires a constant
  // non-negative repeat count, since the result type must be known.
  if (op == "Mult" && (lhs_is_tuple || rhs_is_tuple))
  {
    exprt &tuple = lhs_is_tuple ? lhs : rhs;
    exprt &count = lhs_is_tuple ? rhs : lhs;

    if (
      !count.is_constant() ||
      !(count.type().is_signedbv() || count.type().is_unsignedbv()))
      return nil_exprt();

    BigInt n_big = binary2integer(
      to_constant_expr(count).value().c_str(), count.type().is_signedbv());
    if (n_big < 0)
      n_big = 0;
    const size_t n = n_big.to_int64();

    const auto &components = to_struct_type(tuple.type()).components();

    std::vector<typet> element_types;
    element_types.reserve(components.size() * n);
    for (size_t i = 0; i < n; ++i)
      for (const auto &c : components)
        element_types.push_back(c.type());

    struct_typet new_type =
      tuple_handler_->create_tuple_struct_type(element_types);

    struct_exprt result(new_type);
    // V.3: IREP2 tuple-component access (exact round-trip of member_exprt).
    expr2tc tuple2;
    migrate_expr(tuple, tuple2);
    for (size_t i = 0; i < n; ++i)
      for (const auto &c : components)
        result.copy_to_operands(migrate_expr_back(
          member2tc(migrate_type(c.type()), tuple2, c.get_name())));

    if (element.contains("lineno"))
      result.location() = get_location_from_decl(element);
    return result;
  }

  return nil_exprt();
}

exprt python_converter::handle_list_operations(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right,
  const nlohmann::json &element)
{
  typet list_type = type_handler_.get_list_type();

  // Resolve function calls that return lists to temporary variables
  auto resolve_list_call = [&](exprt &expr) -> bool {
    // Check if this is a side effect function call
    if (expr.id().as_string() != "sideeffect")
      return false;

    if (expr.get("statement") != "function_call")
      return false;

    if (expr.type() != list_type)
      return false;

    locationt location = get_location_from_decl(element);

    // Create temporary variable for the list
    symbolt &tmp_var_symbol = create_tmp_symbol(
      element, "tmp_func_ret", list_type, gen_zero(list_type));

    // Declare the temporary
    code_declt tmp_var_decl(symbol_expr(tmp_var_symbol));
    tmp_var_decl.location() = location;
    current_block->copy_to_operands(tmp_var_decl);

    // Build function call statement from side effect expression
    side_effect_expr_function_callt &side_effect =
      to_side_effect_expr_function_call(expr);

    code_function_callt call;
    call.function() = side_effect.function();
    call.arguments() = side_effect.arguments();
    call.lhs() = symbol_expr(tmp_var_symbol);
    call.type() = list_type;
    call.location() = location;

    current_block->copy_to_operands(call);

    // Replace expr with the temp variable
    expr = symbol_expr(tmp_var_symbol);
    return true;
  };

  // Resolve both sides if they are function calls
  resolve_list_call(lhs);
  resolve_list_call(rhs);

  // List comparison
  if (
    lhs.type() == list_type && rhs.type() == list_type &&
    (op == "Eq" || op == "NotEq" || op == "Lt" || op == "LtE" || op == "Gt" ||
     op == "GtE"))
  {
    python_list list(*this, element);
    return list.compare(lhs, rhs, op);
  }

  // List concatenation: also handle Any-typed (void*) right operand, which
  // occurs when iterating an untyped iterable (e.g. `for r in f()` with no
  // return annotation) and then concatenating with a typed list literal.
  auto is_any_ptr = [](const typet &t) {
    return t.is_pointer() && t.subtype().id() == "empty";
  };
  if (
    lhs.type() == list_type && op == "Add" &&
    (rhs.type() == list_type || is_any_ptr(rhs.type())))
  {
    if (rhs.type() != list_type)
      rhs = typecast_exprt(rhs, list_type);
    python_list list(*this, element);
    return list.build_concat_list_call(lhs, rhs, element);
  }

  // List repetition
  if ((lhs.type() == list_type || rhs.type() == list_type) && op == "Mult")
  {
    const bool lhs_is_list = lhs.type() == list_type;
    const bool rhs_is_list = rhs.type() == list_type;
    // list * list is unsupported (Python raises TypeError); avoid routing to
    // repetition lowering, which expects an integer repeat count.
    if (lhs_is_list && rhs_is_list)
    {
      std::ostringstream msg;
      msg << "TypeError: arithmetic on numeric arrays is not supported "
             "(numpy broadcasting is not modelled)";
      const locationt loc = get_location_from_decl(element);
      if (!loc.is_nil())
        msg << " at " << loc.get_file() << ":" << loc.get_line();
      throw std::runtime_error(msg.str());
    }
    if (is_right)
      return nil_exprt();
    python_list list(*this, element);
    return list.list_repetition(left, right, lhs, rhs);
  }

  return nil_exprt();
}

exprt python_converter::handle_relational_type_mismatches(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &element)
{
  // Single character comparisons (including equality/inequality)
  if (type_utils::is_ordered_comparison(op) || op == "Eq" || op == "NotEq")
  {
    // Special handling. Reject cases where both operands are character arrays (like chr(65) == "A")
    // Todo: we should change the all expression to a correct format in future.
    bool both_arrays = lhs.type().is_array() && rhs.type().is_array();

    // If both operands are strings (including char pointers), skip single-char comparison
    // and let the string comparison path handle it (strcmp).
    bool both_strings = type_utils::is_string_type(lhs.type()) &&
                        type_utils::is_string_type(rhs.type());

    if (!both_arrays && !both_strings)
    {
      exprt char_comp_result =
        string_handler_.handle_single_char_comparison(op, lhs, rhs);
      if (!char_comp_result.is_nil())
        return char_comp_result;
    }
  }

  // Float vs string comparisons
  bool lhs_is_float = lhs.type().is_floatbv();
  bool rhs_is_float = rhs.type().is_floatbv();
  bool lhs_is_str = type_utils::is_string_type(lhs.type());
  bool rhs_is_str = type_utils::is_string_type(rhs.type());

  if ((lhs_is_float && rhs_is_str) || (lhs_is_str && rhs_is_float))
  {
    exprt binary_expr(
      python_frontend::map_operator(op, bool_type()), bool_type());

    locationt loc = get_location_from_decl(element);
    if (loc.is_nil() || loc.get_line().empty())
    {
      if (!lhs.location().is_nil())
        loc = lhs.location();
      else if (!rhs.location().is_nil())
        loc = rhs.location();
    }
    binary_expr.location() = loc;

    return handle_float_vs_string(binary_expr, op);
  }

  // Float vs integer: Python promotes int to float for all comparisons.
  // e.g., 3.0 == 3  →  3.0 == 3.0  (True)
  if (lhs_is_float && (rhs.type().is_signedbv() || rhs.type().is_unsignedbv()))
    rhs = typecast_exprt(rhs, lhs.type());
  else if (
    rhs_is_float && (lhs.type().is_signedbv() || lhs.type().is_unsignedbv()))
    lhs = typecast_exprt(lhs, rhs.type());

  return nil_exprt();
}

exprt python_converter::handle_string_binary_operations(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right,
  const nlohmann::json &element)
{
  std::string lhs_type = type_handler_.type_to_string(lhs.type());
  std::string rhs_type = type_handler_.type_to_string(rhs.type());

  // Infer string types for equality comparisons
  if (
    (op == "Eq" || op == "NotEq") && ((lhs_type.empty() && rhs_type == "str") ||
                                      (rhs_type.empty() && lhs_type == "str")))
  {
    if (lhs_type.empty() && element.contains("left"))
    {
      const auto &lhs_expr = element["left"];
      if (
        (lhs_expr.contains("value") && lhs_expr["value"].is_string()) ||
        (lhs_expr.contains("id") && lhs_expr["id"].is_string()))
        lhs_type = "str";
    }
    else if (
      rhs_type.empty() && element.contains("comparators") &&
      element["comparators"].is_array() && !element["comparators"].empty())
    {
      const auto &rhs_expr = element["comparators"][0];
      if (
        (rhs_expr.contains("value") && rhs_expr["value"].is_string()) ||
        (rhs_expr.contains("id") && rhs_expr["id"].is_string()))
        rhs_type = "str";
    }
  }

  // Check for string literals in Add operations
  if (op == "Add")
  {
    if (
      element.contains("left") && element["left"].contains("value") &&
      element["left"]["value"].is_string())
      lhs_type = "str";

    if (
      element.contains("right") && element["right"].contains("value") &&
      element["right"]["value"].is_string())
      rhs_type = "str";

    if (!lhs_type.empty() || !rhs_type.empty())
    {
      if (lhs_type.empty())
        lhs_type = "str";
      if (rhs_type.empty())
        rhs_type = "str";
    }
  }

  // Check if both operands are strings
  bool lhs_is_string =
    (lhs_type == "str") || type_utils::is_string_type(lhs.type());
  bool rhs_is_string =
    (rhs_type == "str") || type_utils::is_string_type(rhs.type());

  if (
    (lhs_is_string && rhs_is_string) ||
    (op == "Mult" &&
     (lhs_is_string || rhs_is_string || type_utils::is_char_type(lhs.type()) ||
      type_utils::is_char_type(rhs.type()))))
  {
    return string_handler_.handle_string_operations(
      op, lhs, rhs, left, right, element);
  }

  return nil_exprt();
}

exprt python_converter::build_binary_expression(
  const std::string &op,
  exprt &lhs,
  exprt &rhs)
{
  const bool is_bitwise_op = op == "BitAnd" || op == "BitOr" ||
                             op == "BitXor" || op == "LShift" || op == "RShift";

  if (is_bitwise_op)
  {
    const typet target_int = int_type();
    if (lhs.type().is_floatbv() || lhs.type().is_bool())
      lhs = typecast_exprt(lhs, target_int);
    if (rhs.type().is_floatbv() || rhs.type().is_bool())
      rhs = typecast_exprt(rhs, target_int);
  }

  auto is_bv_or_bool = [](const typet &t) {
    return t.is_signedbv() || t.is_unsignedbv() || t.is_bool();
  };
  auto bit_width = [](const typet &t) -> unsigned {
    if (t.is_bool())
      return 1;
    return static_cast<const bv_typet &>(t).get_width();
  };
  // Adjust types for non-relational operations
  if (!type_utils::is_relational_op(op))
  {
    // Check for critical type incompatibilities
    const typet &lhs_type = lhs.type();
    const typet &rhs_type = rhs.type();

    // Check for bitvector width mismatch
    if (
      is_bv_or_bool(lhs_type) && is_bv_or_bool(rhs_type) &&
      (bit_width(lhs_type) != bit_width(rhs_type) ||
       lhs_type.is_signedbv() != rhs_type.is_signedbv()))
    {
      const typet &target_type =
        bit_width(lhs_type) >= bit_width(rhs_type) ? lhs_type : rhs_type;

      if (lhs.type() != target_type)
        lhs = typecast_exprt(lhs, target_type);
      if (rhs.type() != target_type)
        rhs = typecast_exprt(rhs, target_type);
    }
  }
  else if (
    (op == "Eq" || op == "NotEq" || op == "Lt" || op == "LtE" || op == "Gt" ||
     op == "GtE") &&
    is_bv_or_bool(lhs.type()) && is_bv_or_bool(rhs.type()) &&
    bit_width(lhs.type()) != bit_width(rhs.type()))
  {
    // Defensive normalization before SMT encoding: keep both operands with a
    // common bit-width to avoid backend assertion failures in comparisons.
    const unsigned lhs_width = bit_width(lhs.type());
    const unsigned rhs_width = bit_width(rhs.type());
    const unsigned common_width = std::max(lhs_width, rhs_width);
    const bool use_signed =
      lhs.type().is_signedbv() || rhs.type().is_signedbv();
    typet common_type;
    if (use_signed)
      common_type = signedbv_typet(common_width);
    else
      common_type = unsignedbv_typet(common_width);

    lhs = typecast_exprt(lhs, common_type);
    rhs = typecast_exprt(rhs, common_type);
  }

  // Determine result type
  typet type;
  if (type_utils::is_relational_op(op))
    type = bool_type();
  else if (op == "Div" || op == "div")
    type = double_type();
  else if (is_bitwise_op)
    type = lhs.type();
  else if (lhs.type().is_floatbv() || rhs.type().is_floatbv())
    type = lhs.type().is_floatbv() ? lhs.type() : rhs.type();
  else
    type = lhs.type();

  // Create expression
  exprt bin_expr(python_frontend::map_operator(op, type), type);

  // Set location
  if (lhs.is_symbol())
    bin_expr.location() = lhs.location();
  else if (rhs.is_symbol())
    bin_expr.location() = rhs.location();

  // Handle signed/unsigned promotion
  if (lhs.type().is_unsignedbv() && rhs.type().is_signedbv())
    rhs.make_typecast(lhs.type());

  // Handle division promotion
  if (op == "Div" || op == "div")
    math_handler_.handle_float_division(lhs, rhs, bin_expr);

  // Add operands
  bin_expr.copy_to_operands(lhs, rhs);

  return bin_expr;
}

void python_converter::promote_ieee_operands(
  exprt &bin_expr,
  const exprt &lhs,
  const exprt &rhs)
{
  if (!is_ieee_op(bin_expr))
    return;

  const typet &target_type = lhs.type().is_floatbv() ? lhs.type() : rhs.type();

  if (!lhs.type().is_floatbv())
    bin_expr.op0() = typecast_exprt(lhs, target_type);
  if (!rhs.type().is_floatbv())
    bin_expr.op1() = typecast_exprt(rhs, target_type);
}
