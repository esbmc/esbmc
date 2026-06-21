#include <python-frontend/string/char_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/string/string_handler.h>
#include <python-frontend/type_utils.h>
#include <irep2/irep2_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/migrate.h>
#include <util/python_types.h>

namespace
{
// V.3: build a boolean constant in IREP2, back-migrated for the legacy
// callers (the comparison fold results consumed as exprt). Exact migrate of
// gen_bool(v) modulo the cosmetic #cpp_type hint.
exprt gen_bool(bool v)
{
  return migrate_expr_back(v ? gen_true_expr() : gen_false_expr());
}
} // namespace

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
    if (sym && sym->get_value().is_constant())
      resolved_lhs = sym->get_value();
  }

  if (rhs.is_symbol() && rhs.type().is_array())
  {
    const symbolt *sym = symbol_table_.find_symbol(rhs.identifier());
    if (sym && sym->get_value().is_constant())
      resolved_rhs = sym->get_value();
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
    return gen_bool((op == "Eq") ? equal : !equal);
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
        return gen_bool(op == "NotEq");

      // Extract string values
      std::string lhs_str =
        string_handler_.extract_string_from_array_operands(lhs);
      std::string rhs_str =
        string_handler_.extract_string_from_array_operands(rhs);

      // Compare strings
      bool equal = (lhs_str == rhs_str);
      return gen_bool((op == "Eq") ? equal : !equal);
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
      return gen_bool((op == "Eq") ? equal : !equal);
    }
    return gen_bool(op == "NotEq");
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
      return gen_bool((op == "Eq") ? equal : !equal);
    }
    return gen_bool(op == "NotEq");
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
      resolved_array = symbol->get_value();
      if (symbol->get_value().is_symbol())
      {
        const symbolt *compound =
          symbol_table_.find_symbol(symbol->get_value().identifier());
        if (compound && compound->get_value().is_constant())
          resolved_array = compound->get_value();
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
    return gen_bool((op == "Eq") ? strings_equal : !strings_equal);
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
      return gen_bool(op == "NotEq");

    // Both are string arrays: compare based on content
    bool lhs_empty = string_handler_.is_zero_length_array(lhs) ||
                     (lhs.is_constant() && lhs.operands().size() <= 1);
    bool rhs_empty = string_handler_.is_zero_length_array(rhs) ||
                     (rhs.is_constant() && rhs.operands().size() <= 1);

    if (lhs_empty != rhs_empty)
      return gen_bool(op == "NotEq");

    if (lhs.size() != rhs.size())
      return gen_bool(op == "NotEq");

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
    return gen_bool(op == "Eq");

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

    // Build a single-character access as IREP2; the comparison below composes
    // these and back-migrates once at the return boundary (V.3).
    auto char_at_index = [&](const exprt &expr, int idx) -> expr2tc {
      exprt index = from_integer(idx, index_type());

      if (expr.type().is_array())
      {
        // exact round-trip of index_exprt(expr, index, char_type())
        expr2tc expr2, index2;
        migrate_expr(expr, expr2);
        migrate_expr(index, index2);
        return index2tc(migrate_type(char_type()), expr2, index2);
      }

      // Pointer path: reproduce the legacy dereference node verbatim — its
      // two-arg constructor takes char_type().subtype() (nil), not char_type(),
      // as the result type — then migrate it forward for a uniform expr2tc.
      exprt ptr = expr;
      if (!ptr.type().is_pointer())
        ptr = address_of_exprt(expr);
      plus_exprt ptr_plus(ptr, index);
      ptr_plus.type() = ptr.type();
      expr2tc deref2;
      migrate_expr(dereference_exprt(ptr_plus, char_type()), deref2);
      return deref2;
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
      expr2tc first_char = char_at_index(str_expr, 0);
      expr2tc second_char = char_at_index(str_expr, 1);

      expr2tc lit, zero;
      migrate_expr(
        from_integer(static_cast<unsigned char>(literal_char), char_type()),
        lit);
      migrate_expr(from_integer(0, char_type()), zero);

      // (str[0] == lit) and (str[1] == '\0'); negated for NotEq.
      expr2tc both_eq =
        and2tc(equality2tc(first_char, lit), equality2tc(second_char, zero));

      if (op == "NotEq")
      {
        exprt not_expr = migrate_expr_back(not2tc(both_eq));
        not_expr.location() = get_location_from_decl(element);
        return not_expr;
      }

      exprt both_eq_expr = migrate_expr_back(both_eq);
      both_eq_expr.location() = get_location_from_decl(element);
      return both_eq_expr;
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

exprt python_converter::try_lower_slice_member_is_none(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs)
{
  // Restrict to identity comparisons. Value comparisons (`sl.start == None`)
  // fall through to the existing int-vs-None handling, which preserves
  // Python value-equality semantics if the field ever holds a real integer.
  if (op != "Is" && op != "IsNot")
    return nil_exprt();

  // One operand must be the None literal; the other must be a member access
  // on a __ESBMC_PySliceObj struct.
  const exprt *member_side = nullptr;
  if (lhs.type() == none_type() && rhs.id() == "member")
    member_side = &rhs;
  else if (rhs.type() == none_type() && lhs.id() == "member")
    member_side = &lhs;
  else
    return nil_exprt();

  const namespacet ns(symbol_table_);
  const typet base_type = ns.follow(member_side->op0().type());
  if (!base_type.is_struct())
    return nil_exprt();
  if (base_type != ns.follow(type_handler_.get_slice_type()))
    return nil_exprt();

  const struct_typet &slice_struct = to_struct_type(base_type);
  const std::string flag_name =
    "has_" + member_side->component_name().as_string();
  if (!slice_struct.has_component(flag_name))
    return nil_exprt();
  const typet flag_type = slice_struct.get_component(flag_name).type();

  // V.3: build the flag member access and the is/is-not-None check in IREP2,
  // back-migrated once at the return.
  expr2tc fb2;
  migrate_expr(member_side->op0(), fb2);
  const expr2tc flag2 = member2tc(migrate_type(flag_type), fb2, flag_name);
  // `sl.start is None` ⇔ flag is zero (bound was absent).
  // `sl.start is not None` ⇔ flag is non-zero (bound was supplied).
  const expr2tc eq2 = equality2tc(flag2, gen_zero(flag2->type));
  return migrate_expr_back(op == "Is" ? eq2 : not2tc(eq2));
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

  // Bare slice components (`sl.start`, `sl.stop`, `sl.step`) hold
  // nondeterministic values when the user did not supply a bound. The
  // authoritative "was the bound supplied?" signal lives in the companion
  // `has_start` / `has_stop` / `has_step` flags of __ESBMC_PySliceObj.
  // Lower `sl.<field> is None` to `sl.has_<field> == 0` so that user code can
  // distinguish a bare `:` from an explicit `0:0` (github #4543).
  if (exprt rewrite = try_lower_slice_member_is_none(op, lhs, rhs);
      !rewrite.is_nil())
    return rewrite;

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
      return gen_bool(!is_eq);
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
    return gen_bool(op == "NotEq");
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
        if (symbol && symbol->get_value().is_constant())
        {
          std::string val =
            to_constant_expr(symbol->get_value()).get_value().as_string();

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
