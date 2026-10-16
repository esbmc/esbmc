#include <python-frontend/string/char_utils.h>
#include <python-frontend/exception/python_exception_handler.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/lambda/python_lambda.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_expr_builder.h>
#include <python-frontend/string/string_handler.h>
#include <python-frontend/type/type_utils.h>
#include <irep2/irep2_utils.h>
#include <util/arith/arith_tools.h>
#include <util/lang/c_types.h>
#include <util/irep/migrate.h>
#include <util/lang/python_types.h>
#include <set>

namespace
{
// V.3: build a boolean constant in IREP2, back-migrated for the legacy
// callers (the comparison fold results consumed as exprt). Exact migrate of
// gen_bool(v) modulo the cosmetic #cpp_type hint.
exprt gen_bool(bool v)
{
  return migrate_expr_back(v ? gen_true_expr() : gen_false_expr());
}

// A class object shares a string's char-array model but never equals one
// (int != "int"), and orders against nothing.
exprt compare_class_objects(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs)
{
  if (op != "Eq" && op != "NotEq")
    throw std::runtime_error("TypeError: ordered comparison of a class object");
  const bool same_class = python_converter::is_class_object(lhs) &&
                          python_converter::is_class_object(rhs) &&
                          lhs.get("value") == rhs.get("value");
  return gen_bool(same_class == (op == "Eq"));
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

  // Resolve an op plus a three-way comparison sign (cmp < 0 / == 0 / > 0) to a
  // bool literal, or nil for an op this helper does not cover. The ordered
  // operators (Lt/LtE/Gt/GtE) previously collapsed to `!equal`, so ordered
  // comparisons of constant strings/chars were wrong whenever the answer
  // differed from inequality (e.g. "abc" <= "abc" folded to False, and
  // "abd" <= "abc" to True).
  auto resolve = [&op](int cmp) -> exprt {
    if (op == "Eq")
      return gen_bool(cmp == 0);
    if (op == "NotEq")
      return gen_bool(cmp != 0);
    if (op == "Lt")
      return gen_bool(cmp < 0);
    if (op == "LtE")
      return gen_bool(cmp <= 0);
    if (op == "Gt")
      return gen_bool(cmp > 0);
    if (op == "GtE")
      return gen_bool(cmp >= 0);
    return nil_exprt();
  };

  // Characters order by code point (unsigned byte), matching both Python's
  // string ordering and the lexicographic char-array / strcmp paths. Read the
  // byte unsigned so a high-bit char (≥ 128) is not mis-ordered as negative.
  auto char_value = [](const exprt &e) -> BigInt {
    return binary2integer(e.value().as_string(), false);
  };

  // Single character comparisons
  if (
    (lhs.type().is_unsignedbv() || lhs.type().is_signedbv()) &&
    (rhs.type().is_unsignedbv() || rhs.type().is_signedbv()))
  {
    const BigInt l = char_value(lhs);
    const BigInt r = char_value(rhs);
    return resolve(l < r ? -1 : (l > r ? 1 : 0));
  }

  // Array vs array comparisons (string literal comparison)
  if (lhs.type().is_array() && rhs.type().is_array())
  {
    // Check if both are char arrays (strings)
    if (
      lhs.type().subtype() == char_type() &&
      rhs.type().subtype() == char_type())
    {
      if (is_class_object(lhs) || is_class_object(rhs))
        return compare_class_objects(op, lhs, rhs);

      // Extract string values and compare lexicographically (Python orders
      // strings by code point; for the char-array model this is byte-wise,
      // matching the runtime strcmp path).
      std::string lhs_str =
        string_handler_.extract_string_from_array_operands(lhs);
      std::string rhs_str =
        string_handler_.extract_string_from_array_operands(rhs);

      return resolve(lhs_str.compare(rhs_str));
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
      const BigInt l = char_value(lhs);
      const BigInt r = char_value(rhs_ops[0]);
      return resolve(l < r ? -1 : (l > r ? 1 : 0));
    }
    return (op == "Eq" || op == "NotEq") ? gen_bool(op == "NotEq")
                                         : nil_exprt();
  }

  if (
    lhs.type().is_array() &&
    (rhs.type().is_unsignedbv() || rhs.type().is_signedbv()))
  {
    const exprt::operandst &lhs_ops = lhs.operands();
    if (lhs_ops.size() == 1)
    {
      const BigInt l = char_value(lhs_ops[0]);
      const BigInt r = char_value(rhs);
      return resolve(l < r ? -1 : (l > r ? 1 : 0));
    }
    return (op == "Eq" || op == "NotEq") ? gen_bool(op == "NotEq")
                                         : nil_exprt();
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

static bool either_is_string(const exprt &lhs, const exprt &rhs)
{
  return type_utils::is_string_type(lhs.type()) ||
         type_utils::is_string_type(rhs.type());
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
  // This avoids introducing strcmp() calls that can inflate branch coverage
  // counts.
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

      // Pointer path: build the pointer-plus-index arithmetic and the
      // dereference natively in IREP2 (V.1k arith site). The legacy two-arg
      // dereference_exprt(op, tp) ctor sets the result type to tp.subtype()
      // (nil, since char_type() is not a pointer type) rather than tp
      // itself; migrate_type(char_type().subtype()) reproduces that exact
      // quirk byte-for-byte.
      exprt ptr = expr;
      if (!ptr.type().is_pointer())
        ptr = address_of_exprt(expr);
      expr2tc ptr2, index2;
      migrate_expr(ptr, ptr2);
      migrate_expr(index, index2);
      expr2tc ptr_plus2 = add2tc(ptr2->type, ptr2, index2);
      return dereference2tc(migrate_type(char_type().subtype()), ptr_plus2);
    };

    char literal_char = 0;
    bool lhs_is_char_literal = extract_single_char(resolved_lhs, literal_char);
    bool rhs_is_char_literal = extract_single_char(resolved_rhs, literal_char);

    const exprt &str_expr = lhs_is_char_literal ? resolved_rhs : resolved_lhs;
    exprt reread;
    exprt first_read = lhs_is_char_literal ^ rhs_is_char_literal &&
                           either_is_string(resolved_lhs, resolved_rhs)
                         ? bind_operand_read_twice(str_expr, element, reread)
                         : nil_exprt();
    if (first_read.is_not_nil())
    {
      expr2tc first_char = char_at_index(first_read, 0);
      expr2tc second_char = char_at_index(reread, 1);

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

  // V.3: build the strcmp() expression-context call in IREP2. The string base
  // addresses are clean pointers (built via build_index/build_address_of), so
  // this is the byte-identical round-trip of the legacy call node.
  exprt strcmp_call = python_expr::build_call_expr(
    *strncmp_symbol, int_type(), {resolved_lhs, resolved_rhs});
  strcmp_call.location() = get_location_from_decl(element);

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

/// Operators that CPython rejects outright when either operand is None:
/// the arithmetic and ordering ones. Identity and equality are defined on
/// None, and membership takes None as an element (`None in [1, 2]` is False),
/// so both are absent here by design.
static bool raises_on_none_operand(const std::string &op)
{
  static const std::set<std::string> raising{
    "Add",
    "Sub",
    "Mult",
    "Div",
    "FloorDiv",
    "Mod",
    "Pow",
    "LShift",
    "RShift",
    "BitOr",
    "BitXor",
    "BitAnd",
    "MatMult",
    "Lt",
    "LtE",
    "Gt",
    "GtE"};

  return raising.count(op) != 0;
}

/// Python source spelling of an AST operator name, for diagnostics that quote
/// the operator the way CPython does. Falls back to the AST name.
static std::string python_operator_symbol(const std::string &op)
{
  static const std::map<std::string, std::string> symbols{
    {"Add", "+"},
    {"Sub", "-"},
    {"Mult", "*"},
    {"Div", "/"},
    {"FloorDiv", "//"},
    {"Mod", "%"},
    {"Pow", "**"},
    {"LShift", "<<"},
    {"RShift", ">>"},
    {"BitOr", "|"},
    {"BitXor", "^"},
    {"BitAnd", "&"},
    {"MatMult", "@"}};

  auto it = symbols.find(op);
  return it == symbols.end() ? op : it->second;
}

/// Entry point for a binary operation with a None operand. Rejects the
/// operators CPython rejects, and otherwise defers to the identity/equality
/// handling. Kept separate from both callers so neither grows a decision
/// point: get_binary_operator_expr is already far over the complexity gate.
exprt python_converter::handle_none_operand(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs)
{
  if (raises_on_none_operand(op))
    return get_exception_handler().gen_exception_raise(
      "TypeError",
      "unsupported operand type(s) for " + python_operator_symbol(op) + ": '" +
        type_handler_.get_python_type_name(lhs.type()) + "' and '" +
        type_handler_.get_python_type_name(rhs.type()) + "'");

  return handle_none_comparison(op, lhs, rhs);
}

/// Optional counterpart of handle_none_operand: whether an operand is None is
/// only known at run time, so the TypeError is raised under its flag (#8248).
void python_converter::raise_on_optional_none(
  const std::string &op,
  const exprt &lhs_none,
  const exprt &rhs_none,
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &element)
{
  if (!raises_on_none_operand(op))
    return;
  const locationt location = get_location_from_decl(element);
  auto raise = [&](
                 const exprt &flag,
                 const std::string &lhs_name,
                 const std::string &rhs_name) {
    if (flag.is_false())
      return;
    emit_guarded_raise(
      flag,
      "TypeError",
      "unsupported operand type(s) for " + python_operator_symbol(op) + ": '" +
        lhs_name + "' and '" + rhs_name + "'",
      location);
  };
  raise(lhs_none, "NoneType", type_handler_.get_python_type_name(rhs.type()));
  raise(rhs_none, type_handler_.get_python_type_name(lhs.type()), "NoneType");
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
  if (
    exprt rewrite = try_lower_slice_member_is_none(op, lhs, rhs);
    !rewrite.is_nil())
    return rewrite;

  // If one side is None and the other is a different type (e.g., int, str)
  // Handle type mismatch comparison
  if (lhs_is_none != rhs_is_none)
  {
    // Check if the non-None side is a different type
    const exprt &non_none = lhs_is_none ? rhs : lhs;

    // If the non-none side is an Optional struct, check its is_none field
    // directly
    if (type_utils::is_optional_struct(non_none.type()))
    {
      exprt flag = member_exprt(non_none, "is_none", bool_type());
      return is_eq ? flag : not_exprt(flag);
    }

    // If comparing with a constant integer, string, or other non-None constant
    // exclude pointer to array (strings), as they could be Optional[str]
    // parameters
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

  // Create isnone expression with unwrapped operands.
  // V.3: build the isnone node in IREP2. isnone2t is a 2-operand custom kind
  // with forward/back migrate arms (since #3289), so migrating the operands,
  // building, and back-migrating once is the exact round-trip of the legacy
  // node. Spike-1 confirmed the operands are already resolved at this site.
  expr2tc lhs2, rhs2;
  migrate_expr(lhs, lhs2);
  migrate_expr(rhs, rhs2);
  exprt isnone_expr = migrate_expr_back(isnone2tc(lhs2, rhs2));

  // If checking inequality, wrap with not
  if (!is_eq)
  {
    exprt not_expr("not", typet("bool"));
    not_expr.move_to_operands(isnone_expr);
    return not_expr;
  }

  return isnone_expr;
}

/// The module-level name that owns the object `node` denotes, or "" when that
/// is not provable from the AST. Each name on the way must be bound exactly
/// once in the whole module, by a top-level statement, and every name but the
/// owner by `name = other_name`. Such a name holds one object for the whole
/// run, so two names with one owner are the same object, whatever the value
/// is modelled as: a str is copied on assignment (#8412).
static std::string
sole_object_owner(const nlohmann::json &module, const nlohmann::json &node)
{
  std::set<std::string> seen;
  for (const nlohmann::json *name = &node; name->value("_type", "") == "Name";)
  {
    const std::string id = (*name)["id"];
    const nlohmann::json *value = json_utils::top_level_value(module, id);
    if (
      !value || json_utils::count_name_bindings(module, id) != 1 ||
      !seen.insert(id).second)
      return "";
    if (value->value("_type", "") != "Name")
      return id;
    name = value;
  }
  return "";
}

enum class identity_kind
{
  function,
  instance,
  other
};

/// A code-typed operand is the def itself only when it is a symbol: a call's
/// result merely carries the code type.
static identity_kind
identity_kind_of(const expr2tc &operand, bool user_class_pointer)
{
  if (
    (is_code_type(operand->type) && is_symbol2t(operand)) ||
    (is_pointer_type(operand->type) &&
     is_code_type(to_pointer_type(operand->type).subtype)))
    return identity_kind::function;
  return user_class_pointer ? identity_kind::instance : identity_kind::other;
}

/// `is` with a function on one side, or nil when neither side is a function.
/// A def is a code symbol and a name bound to one a pointer to code, so both
/// compare as addresses and the pointer model decides identity on each path.
/// A class instance is never a function: comparing its address gives False
/// and keeps any call inside the operand. Any other operand may be a function
/// that lost its type in a container, so it is left to the caller.
static expr2tc function_identity(
  const expr2tc &lhs,
  const expr2tc &rhs,
  identity_kind lhs_kind,
  identity_kind rhs_kind)
{
  if (
    lhs_kind == identity_kind::other || rhs_kind == identity_kind::other ||
    (lhs_kind == identity_kind::instance &&
     rhs_kind == identity_kind::instance))
    return expr2tc();
  const type2tc ptr = pointer_type2tc(get_empty_type());
  auto address = [&ptr](const expr2tc &e) {
    return typecast2tc(
      ptr, is_code_type(e->type) ? address_of2tc(e->type, e) : e);
  };
  return equality2tc(address(lhs), address(rhs));
}

/// Build the Python 'is' identity equality in IREP2. The operands are the
/// raw values, or the arrays' base addresses when both sides are arrays.
expr2tc python_converter::build_is_equality(
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right)
{
  const std::string owner = sole_object_owner(ast(), left);
  if (!owner.empty() && owner == sole_object_owner(ast(), right))
    return gen_true_expr();

  expr2tc lhs2, rhs2;
  if (lhs.type().is_array() && rhs.type().is_array())
  {
    // Compare base addresses of the arrays
    migrate_expr(string_handler_.get_array_base_address(lhs), lhs2);
    migrate_expr(string_handler_.get_array_base_address(rhs), rhs2);
  }
  else
  {
    // Default identity comparison
    migrate_expr(lhs, lhs2);
    migrate_expr(rhs, rhs2);
  }

  if (
    expr2tc fn = function_identity(
      lhs2,
      rhs2,
      identity_kind_of(lhs2, is_user_class_pointer(lhs.type())),
      identity_kind_of(rhs2, is_user_class_pointer(rhs.type())));
    !is_nil_expr(fn))
    return fn;

  /* An object that lost its type through a container comes back as an
   * integer, and comparing that with the object's pointer builds an equality
   * every backend's mk_eq rejects on the width mismatch (#6640 territory).
   * A function that is not known to be a def is the same case. Say so here
   * rather than abort in the solver. */
  if (
    is_pointer_type(lhs2->type) != is_pointer_type(rhs2->type) ||
    ((is_code_type(lhs2->type) || is_code_type(rhs2->type)) &&
     !is_array_type(lhs2->type) && !is_array_type(rhs2->type)))
    throw std::runtime_error(
      "'is' with an operand that lost its object type, which ESBMC cannot "
      "compare by identity");

  return equality2tc(lhs2, rhs2);
}

/// Construct the expression for Python 'is' operator
exprt python_converter::get_binary_operator_expr_for_is(
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right)
{
  return migrate_expr_back(build_is_equality(lhs, rhs, left, right));
}

/// Construct the negation of an 'is' expression, used for 'is not'
exprt python_converter::get_negated_is_expr(
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right)
{
  return migrate_expr_back(not2tc(build_is_equality(lhs, rhs, left, right)));
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
    // Python allows this comparison but it always returns False for Eq and True
    // for NotEq For verification purposes, we model this as returning the
    // expected constant value This represents Python's behavior: str == int
    // always evaluates to False
    return gen_bool(op == "NotEq");
  }

  return nil_exprt(); // No action taken for other operators
}

/// The argument of @p node when it is `type(x)` on a name whose static type
/// was erased to `void *`; nil otherwise.
exprt python_converter::erased_type_argument(const nlohmann::json &node)
{
  if (
    node["_type"] != "Call" || node["func"]["_type"] != "Name" ||
    node["func"]["id"] != "type" || node["args"].size() != 1 ||
    node["args"][0]["_type"] != "Name")
    return nil_exprt();

  exprt value = get_expr(node["args"][0]);
  if (!value.type().is_pointer() || value.type().subtype().id() != "empty")
    return nil_exprt();
  return value;
}

exprt python_converter::handle_erased_type_comparison(
  const std::string &op,
  const nlohmann::json &left,
  const nlohmann::json &right)
{
  if (op != "Is" && op != "IsNot" && op != "Eq" && op != "NotEq")
    return nil_exprt();

  static const std::set<std::string> scalars = {"int", "float", "bool", "str"};
  auto scalar_name = [](const nlohmann::json &n) -> std::string {
    return n["_type"] == "Name" && scalars.count(n["id"].get<std::string>())
             ? n["id"].get<std::string>()
             : "";
  };
  const bool type_on_right = !scalar_name(right).empty();
  const std::string name = scalar_name(type_on_right ? right : left);
  if (name.empty())
    return nil_exprt();
  const exprt value = erased_type_argument(type_on_right ? left : right);
  if (value.is_nil())
    return nil_exprt();

  // The static type says nothing, so let symex resolve the object as it does
  // for isinstance(), refusing when it cannot. bool is int's only builtin
  // subclass.
  auto isinstance_of = [&](const std::string &type_name) {
    expr2tc value2, type2;
    migrate_expr(value, value2);
    migrate_expr(gen_zero(type_handler_.get_typet(type_name, 0)), type2);
    return migrate_expr_back(isinstance2tc(value2, type2));
  };
  exprt result = isinstance_of(name);
  if (name == "int")
    result = and_exprt(result, not_exprt(isinstance_of("bool")));
  return op == "Is" || op == "Eq" ? result : not_exprt(result);
}

exprt python_converter::handle_type_comparison(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right)
{
  const exprt erased = handle_erased_type_comparison(op, left, right);
  return erased.is_nil() ? handle_type_identity_check(op, lhs, rhs, left, right)
                         : erased;
}

static bool is_type_call(const nlohmann::json &node)
{
  return node["_type"] == "Call" && node["func"]["_type"] == "Name" &&
         node["func"]["id"] == "type";
}

std::optional<std::string> python_converter::class_identity_operand(
  const nlohmann::json &node,
  const exprt &expr) const
{
  // type(x) is T (GitHub #5936). An unrecognised name leaves this side
  // unresolved, so the fold stays as conservative as it was.
  if (is_type_call(node))
  {
    if (!expr.is_constant())
      return std::nullopt;
    const std::string name = expr.get_string("value");
    return type_utils::is_type_identifier(name) ? std::optional(name)
                                                : std::nullopt;
  }

  if (is_class_object(expr))
    return expr.get_string("value");

  if (node["_type"] != "Name" || !expr.is_symbol())
    return std::nullopt;
  const std::string *held = class_object_name(expr.identifier());
  if (!held)
    return std::nullopt;
  const std::string name = node["id"].get<std::string>();
  if (count_bindings(*ast_json, name) != 1)
    throw std::runtime_error(
      "Unsupported: identity of '" + name +
      "', which holds a class and is bound more than once");
  return *held;
}

exprt python_converter::handle_type_identity_check(
  const std::string &op,
  const exprt &lhs,
  const exprt &rhs,
  const nlohmann::json &left,
  const nlohmann::json &right)
{
  if (op != "Is" && op != "IsNot")
    return nil_exprt();

  const std::optional<std::string> lhs_class =
    class_identity_operand(left, lhs);
  const std::optional<std::string> rhs_class =
    class_identity_operand(right, rhs);
  // An unresolved type(x) still denotes a class, so it is left to the
  // comparison paths that follow.
  if (
    (!lhs_class && !rhs_class) || (!lhs_class && is_type_call(left)) ||
    (!rhs_class && is_type_call(right)))
    return nil_exprt();

  // A value is never identical to a class object.
  const bool same = lhs_class && rhs_class && *lhs_class == *rhs_class;
  return gen_bool(same == (op == "Is"));
}
