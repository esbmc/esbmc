#include <python-frontend/char_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/string_handler.h>
#include <python-frontend/type_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/python_types.h>

std::pair<exprt, exprt> python_converter::resolve_comparison_operands_internal(
  const exprt &lhs,
  const exprt &rhs)
{
  exprt resolved_lhs = lhs;
  exprt resolved_rhs = rhs;

  // Only resolve constant arrays, not pointers
  if (lhs.is_symbol() && lhs.type().is_array())
  {
    const symbolt *sym = symbol_table_.find_symbol(lhs.identifier());
    if (sym && sym->value.is_constant())
      resolved_lhs = sym->value;
  }

  if (rhs.is_symbol() && rhs.type().is_array())
  {
    const symbolt *sym = symbol_table_.find_symbol(rhs.identifier());
    if (sym && sym->value.is_constant())
      resolved_rhs = sym->value;
  }

  return {resolved_lhs, resolved_rhs};
}

bool python_converter::has_unsupported_side_effects_internal(
  const exprt &lhs,
  const exprt &rhs)
{
  auto has_unsupported_side_effect = [](const exprt &expr) {
    return expr.id() == "sideeffect" &&
           expr.get("statement") != "function_call";
  };

  return has_unsupported_side_effect(lhs) || has_unsupported_side_effect(rhs);
}

exprt python_converter::compare_constants_internal(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs)
{
  if (!lhs.is_constant() || !rhs.is_constant())
    return nil_exprt();

  // Single character comparisons
  if (
    (lhs.type().is_unsignedbv() || lhs.type().is_signedbv()) &&
    (rhs.type().is_unsignedbv() || rhs.type().is_signedbv()))
  {
    bool equal = (lhs == rhs);
    return gen_boolean((op == "Eq") ? equal : !equal);
  }

  // Array vs array comparisons (string literal comparison)
  if (lhs.type().is_array() && rhs.type().is_array())
  {
    // Check if both are char arrays (strings)
    if (
      lhs.type().subtype() == char_type() &&
      rhs.type().subtype() == char_type())
    {
      // Type-identifier constants (e.g. from `x = int`) have no operands and
      // store the name in get_value(). String literals have individual char
      // operands and an empty get_value(). These represent different Python
      // objects (int != "int"), so comparing across formats is always unequal.
      bool lhs_is_type_id = lhs.operands().empty();
      bool rhs_is_type_id = rhs.operands().empty();
      if (lhs_is_type_id != rhs_is_type_id)
        return gen_boolean(op == "NotEq");

      // Extract string values
      std::string lhs_str =
        string_handler_.extract_string_from_array_operands(lhs);
      std::string rhs_str =
        string_handler_.extract_string_from_array_operands(rhs);

      // Compare strings
      bool equal = (lhs_str == rhs_str);
      return gen_boolean((op == "Eq") ? equal : !equal);
    }
  }

  // Mixed character vs array comparisons
  if (
    (lhs.type().is_unsignedbv() || lhs.type().is_signedbv()) &&
    rhs.type().is_array())
  {
    const exprt::operandst &rhs_ops = rhs.operands();
    if (rhs_ops.size() == 1)
    {
      bool equal =
        (lhs == rhs_ops[0]) || (lhs.get("value") == rhs_ops[0].get("value"));
      return gen_boolean((op == "Eq") ? equal : !equal);
    }
    return gen_boolean(op == "NotEq");
  }

  if (
    lhs.type().is_array() &&
    (rhs.type().is_unsignedbv() || rhs.type().is_signedbv()))
  {
    const exprt::operandst &lhs_ops = lhs.operands();
    if (lhs_ops.size() == 1)
    {
      bool equal =
        (lhs_ops[0] == rhs) || (lhs_ops[0].get("value") == rhs.get("value"));
      return gen_boolean((op == "Eq") ? equal : !equal);
    }
    return gen_boolean(op == "NotEq");
  }

  return nil_exprt();
}

exprt python_converter::handle_indexed_comparison_internal(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs)
{
  if (lhs.id() != "index" || !rhs.is_constant() || !rhs.type().is_array())
    return nil_exprt();

  const exprt &index = lhs.operands()[1];
  BigInt idx =
    binary2integer(index.value().as_string(), index.type().is_signedbv());

  std::string rhs_str = string_handler_.extract_string_from_array_operands(rhs);

  const exprt &array = lhs.operands()[0];
  exprt resolved_array = get_resolved_value(array);

  if (resolved_array.is_nil() && array.is_symbol())
  {
    const symbolt *symbol = symbol_table_.find_symbol(array.identifier());
    if (symbol)
    {
      resolved_array = symbol->value;
      if (symbol->value.is_symbol())
      {
        const symbolt *compound =
          symbol_table_.find_symbol(symbol->value.identifier());
        if (compound && compound->value.is_constant())
          resolved_array = compound->value;
      }
    }
  }

  if (
    !resolved_array.is_nil() && resolved_array.is_constant() &&
    resolved_array.type().is_array() && idx >= 0 &&
    idx < (BigInt)resolved_array.operands().size())
  {
    const exprt &string_element = resolved_array.operands()[idx.to_uint64()];
    std::string lhs_str =
      string_handler_.extract_string_from_array_operands(string_element);
    bool strings_equal = (lhs_str == rhs_str);
    return gen_boolean((op == "Eq") ? strings_equal : !strings_equal);
  }

  return nil_exprt();
}

exprt python_converter::handle_type_mismatches(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs)
{
  // Skip if either operand is a member expression
  if (lhs.is_member() || rhs.is_member())
    return nil_exprt();

  // Check if both are string types (either array or pointer to char)
  bool lhs_is_string =
    (lhs.type().is_array() && lhs.type().subtype() == char_type()) ||
    (lhs.type().is_pointer() && lhs.type().subtype() == char_type());
  bool rhs_is_string =
    (rhs.type().is_array() && rhs.type().subtype() == char_type()) ||
    (rhs.type().is_pointer() && rhs.type().subtype() == char_type());

  // If both are strings (regardless of array vs pointer), let strcmp handle it
  if (lhs_is_string && rhs_is_string)
    return nil_exprt();

  // Types match exactly
  if (lhs.type() == rhs.type())
    return nil_exprt();

  // Both operands are arrays - need to distinguish between lists and strings
  if (lhs.type().is_array() && rhs.type().is_array())
  {
    // Check if these are different semantic types (list vs string)
    bool lhs_is_string_array = (lhs.type().subtype() == char_type());
    bool rhs_is_string_array = (rhs.type().subtype() == char_type());

    // If one is a string array and the other is not, they're different types
    if (lhs_is_string_array != rhs_is_string_array)
      return gen_boolean(op == "NotEq");

    // Both are string arrays: compare based on content
    bool lhs_empty = string_handler_.is_zero_length_array(lhs) ||
                     (lhs.is_constant() && lhs.operands().size() <= 1);
    bool rhs_empty = string_handler_.is_zero_length_array(rhs) ||
                     (rhs.is_constant() && rhs.operands().size() <= 1);

    if (lhs_empty != rhs_empty)
      return gen_boolean(op == "NotEq");

    if (lhs.size() != rhs.size())
      return gen_boolean(op == "NotEq");

    return nil_exprt();
  }

  // Mixed types (array vs non-array, but not both strings)
  // Let strcmp handle the comparison if they're both strings
  return nil_exprt();
}

exprt python_converter::handle_string_comparison(
  const std::string &op,
  exprt &lhs,
  exprt &rhs,
  const nlohmann::json &element)
{
  // Resolve symbols to their constant values
  auto [resolved_lhs, resolved_rhs] =
    resolve_comparison_operands_internal(lhs, rhs);

  // Check for unsupported side effects
  if (has_unsupported_side_effects_internal(resolved_lhs, resolved_rhs))
    throw std::runtime_error("Cannot compare non-function side effects");

  // Handle zero-length arrays early
  if (
    string_handler_.is_zero_length_array(resolved_lhs) &&
    string_handler_.is_zero_length_array(resolved_rhs))
    return gen_boolean(op == "Eq");

  // Fast-path for comparisons against single-character string literals.
  // This avoids introducing strcmp() calls that can inflate branch coverage counts.
  if (op == "Eq" || op == "NotEq")
  {
    auto extract_single_char = [&](const exprt &expr, char &ch) -> bool {
      const exprt *candidate = &expr;
      if (
        expr.id() == "address_of" && expr.operands().size() == 1 &&
        expr.op0().type().is_array())
      {
        candidate = &expr.op0();
      }

      if (!candidate->type().is_array())
        return false;

      if (candidate->type().subtype() != char_type())
        return false;

      const std::string value =
        string_handler_.extract_string_from_array_operands(*candidate);
      if (value.size() != 1)
        return false;

      ch = value[0];
      return true;
    };

    auto char_at_index = [&](const exprt &expr, int idx) -> exprt {
      exprt index = from_integer(idx, index_type());
      if (expr.type().is_array())
        return index_exprt(expr, index, char_type());

      exprt ptr = expr;
      if (!ptr.type().is_pointer())
        ptr = address_of_exprt(expr);

      plus_exprt ptr_plus(ptr, index);
      ptr_plus.type() = ptr.type();
      return dereference_exprt(ptr_plus, char_type());
    };

    char literal_char = 0;
    bool lhs_is_char_literal = extract_single_char(resolved_lhs, literal_char);
    bool rhs_is_char_literal = extract_single_char(resolved_rhs, literal_char);

    if (
      lhs_is_char_literal ^ rhs_is_char_literal &&
      (type_utils::is_string_type(resolved_lhs.type()) ||
       type_utils::is_string_type(resolved_rhs.type())))
    {
      const exprt &str_expr = lhs_is_char_literal ? resolved_rhs : resolved_lhs;
      exprt first_char = char_at_index(str_expr, 0);
      exprt second_char = char_at_index(str_expr, 1);

      exprt lit =
        from_integer(static_cast<unsigned char>(literal_char), char_type());
      exprt zero = from_integer(0, char_type());

      exprt first_eq("=", bool_type());
      first_eq.copy_to_operands(first_char, lit);
      exprt second_eq("=", bool_type());
      second_eq.copy_to_operands(second_char, zero);

      exprt both_eq("and", bool_type());
      both_eq.copy_to_operands(first_eq, second_eq);

      if (op == "NotEq")
      {
        exprt not_expr("not", bool_type());
        not_expr.copy_to_operands(both_eq);
        not_expr.location() = get_location_from_decl(element);
        return not_expr;
      }

      both_eq.location() = get_location_from_decl(element);
      return both_eq;
    }
  }

  // Try constant comparisons
  exprt constant_result =
    compare_constants_internal(op, resolved_lhs, resolved_rhs);
  if (!constant_result.is_nil())
    return constant_result;

  // Try indexed string comparison
  exprt indexed_result =
    handle_indexed_comparison_internal(op, resolved_lhs, resolved_rhs);
  if (!indexed_result.is_nil())
    return indexed_result;

  // Handle type mismatches
  exprt mismatch_result =
    handle_type_mismatches(op, resolved_lhs, resolved_rhs);
  if (!mismatch_result.is_nil())
    return mismatch_result;

  // At this point, both operands should be strings (arrays of char)
  if (resolved_lhs.type().is_array())
    resolved_lhs = string_handler_.get_array_base_address(resolved_lhs);
  if (resolved_rhs.type().is_array())
    resolved_rhs = string_handler_.get_array_base_address(resolved_rhs);

  symbolt *strncmp_symbol = symbol_table_.find_symbol("c:@F@strcmp");
  if (!strncmp_symbol)
    throw std::runtime_error(
      "strcmp function not found in symbol table for string comparison");

  side_effect_expr_function_callt strcmp_call;
  strcmp_call.function() = symbol_expr(*strncmp_symbol);
  strcmp_call.arguments() = {resolved_lhs, resolved_rhs};
  strcmp_call.location() = get_location_from_decl(element);
  strcmp_call.type() = int_type();

  lhs = strcmp_call;
  rhs = gen_zero(int_type());

  return nil_exprt(); // continue with lhs OP rhs
}

exprt python_converter::handle_none_comparison(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs)
{
  const bool is_eq = (op == "Eq" || op == "Is");
  const bool lhs_is_none = (lhs.type() == none_type());
  const bool rhs_is_none = (rhs.type() == none_type());

  // Only handle actual None comparisons
  if (!lhs_is_none && !rhs_is_none)
    return exprt();

  // If one side is None and the other is a different type (e.g., int, str)
  // Handle type mismatch comparison
  if (lhs_is_none != rhs_is_none)
  {
    // Check if the non-None side is a different type
    const exprt &non_none = lhs_is_none ? rhs : lhs;

    // If comparing with a constant integer, string, or other non-None constant
    // exclude pointer to array (strings), as they could be Optional[str] parameters
    if (
      non_none.is_constant() && (!non_none.type().is_pointer() ||
                                 (non_none.type().is_pointer() &&
                                  non_none.type().subtype() != bool_type() &&
                                  non_none.type().subtype() != empty_typet() &&
                                  !non_none.type().subtype().is_array())))
    {
      // None is never equal to non-None constant values
      // For == or is: return False
      // For != or is not: return True
      return gen_boolean(!is_eq);
    }
  }

  // Create isnone expression with unwrapped operands
  exprt isnone_expr("isnone", typet("bool"));
  isnone_expr.copy_to_operands(lhs);
  isnone_expr.copy_to_operands(rhs);

  // If checking inequality, wrap with not
  if (!is_eq)
  {
    exprt not_expr("not", typet("bool"));
    not_expr.move_to_operands(isnone_expr);
    return not_expr;
  }

  return isnone_expr;
}

/// Construct the expression for Python 'is' operator
exprt python_converter::get_binary_operator_expr_for_is(
  const exprt &lhs,
  const exprt &rhs)
{
  typet bool_type_result = bool_type();
  exprt is_expr("=", bool_type_result);

  if (lhs.type().is_array() && rhs.type().is_array())
  {
    // Compare base addresses of the arrays
    is_expr.copy_to_operands(
      string_handler_.get_array_base_address(lhs),
      string_handler_.get_array_base_address(rhs));
  }
  else
  {
    // Default identity comparison
    is_expr.copy_to_operands(lhs, rhs);
  }

  return is_expr;
}

/// Construct the negation of an 'is' expression, used for 'is not'
exprt python_converter::get_negated_is_expr(const exprt &lhs, const exprt &rhs)
{
  exprt is_expr = get_binary_operator_expr_for_is(lhs, rhs);
  exprt not_expr("not", bool_type());
  not_expr.copy_to_operands(is_expr);
  return not_expr;
}

exprt python_converter::handle_string_type_mismatch(
  const exprt &lhs,
  const exprt &rhs,
  const std::string &op)
{
  bool lhs_is_string = type_utils::is_string_type(lhs.type());
  bool rhs_is_string = type_utils::is_string_type(rhs.type());

  // Check if we have a type mismatch
  if (!((lhs_is_string && !rhs_is_string) || (!lhs_is_string && rhs_is_string)))
    return nil_exprt(); // No mismatch, return nil to indicate no action taken

  // Bail out on void* vs string;
  // the caller's strcmp path handles it instead of folding to a static False
  // (the void* may hold a string).
  auto is_void_ptr = [](const typet &t) {
    return t.is_pointer() && t.subtype().id() == "empty";
  };
  if (is_void_ptr(lhs.type()) || is_void_ptr(rhs.type()))
    return nil_exprt();

  exprt lhs_char_value = python_char_utils::get_char_value_as_int(lhs, false);
  exprt rhs_char_value = python_char_utils::get_char_value_as_int(rhs, false);

  if (!lhs_char_value.is_nil() && !rhs_char_value.is_nil())
  {
    return string_handler_.create_char_comparison_expr(
      op, lhs_char_value, rhs_char_value, lhs, rhs);
  }

  // Handle equality/inequality comparisons for other type mismatches
  if (op == "Eq" || op == "NotEq")
  {
    // Python allows this comparison but it always returns False for Eq and True for NotEq
    // For verification purposes, we model this as returning the expected constant value
    // This represents Python's behavior: str == int always evaluates to False
    return gen_boolean(op == "NotEq");
  }

  return nil_exprt(); // No action taken for other operators
}

exprt python_converter::handle_type_identity_check(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right)
{
  // Only handle identity operators
  if (op != "Is" && op != "IsNot")
    return nil_exprt();

  // Resolve type identifiers from either direct names or symbol values
  auto resolve_type_identifier = [&](
                                   const nlohmann::json &node,
                                   const exprt &expr,
                                   std::string &out_name) -> bool {
    if (node["_type"] == "Name" && node.contains("id"))
    {
      std::string name = node["id"].get<std::string>();
      // Check if it's a direct type identifier (e.g., int, str, float)
      if (type_utils::is_type_identifier(name))
      {
        out_name = name;
        return true;
      }
      // Check if it's a variable holding a type object (e.g., x = int)
      if (expr.is_symbol())
      {
        const symbol_exprt &sym = to_symbol_expr(expr);
        const symbolt *symbol = ns.lookup(sym.get_identifier());
        if (symbol && symbol->value.is_constant())
        {
          std::string val =
            to_constant_expr(symbol->value).get_value().as_string();

          if (type_utils::is_type_identifier(val))
          {
            out_name = val;
            return true;
          }
        }
      }
    }

    return false;
  };

  std::string lhs_type_name;
  std::string rhs_type_name;
  bool lhs_is_type = resolve_type_identifier(left, lhs, lhs_type_name);
  bool rhs_is_type = resolve_type_identifier(right, rhs, rhs_type_name);

  // If neither operand is a type identifier, not a type identity check
  if (!lhs_is_type && !rhs_is_type)
    return nil_exprt();

  // If both are type identifiers, compare them
  if (lhs_is_type && rhs_is_type)
  {
    bool same_type = (lhs_type_name == rhs_type_name);
    if (op == "Is")
    {
      if (same_type)
        return true_exprt();
      else
        return false_exprt();
    }
    else // op == "IsNot"
    {
      if (same_type)
        return false_exprt();
      else
        return true_exprt();
    }
  }

  // If only one side is a type identifier, they can never be identical
  // (a value can't be identical to a type object)
  if (op == "Is")
    return false_exprt();
  else // op == "IsNot"
    return true_exprt();
}
