#include "python_dict_internal.h"

using namespace python_expr;

exprt python_dict_handler::compare(
  const exprt &lhs,
  const exprt &rhs,
  const std::string &op)
{
  locationt location = lhs.location();
  if (location.is_nil())
    location = rhs.location();

  typet list_type = type_handler_.get_list_type();

  // Get keys and values from both dicts
  exprt lhs_keys = build_member(lhs, "keys", list_type);
  exprt lhs_values = build_member(lhs, "values", list_type);
  exprt rhs_keys = build_member(rhs, "keys", list_type);
  exprt rhs_values = build_member(rhs, "values", list_type);

  // Find __ESBMC_dict_eq function
  const symbolt *dict_eq_func =
    symbol_table_.find_symbol("c:@F@__ESBMC_dict_eq");

  if (!dict_eq_func)
    throw std::runtime_error("__ESBMC_dict_eq not found in symbol table");

  // Create temp for result
  symbolt &result_var = converter_.create_tmp_symbol(
    nlohmann::json(), "$dict_eq_result$", bool_type(), exprt());
  code_declt result_decl(build_symbol(result_var));
  result_decl.location() = location;
  converter_.add_instruction(result_decl);

  // float_type_id: the runtime type_id Python's `float` is stamped with
  // everywhere (see get_list_element_info in list_mutation.cpp), so
  // __ESBMC_dict_eq can numerically compare an int value against a float
  // value (Python's `1 == 1.0`) instead of rejecting them on a type_id byte
  // mismatch. This is a single global constant, not per-dict state -- float
  // always lowers to the same type, unlike list's type map which also has to
  // distinguish string/int for its own flags.
  const size_t float_type_id =
    std::hash<std::string>{}(type_handler_.type_to_string(
      type_handler_.get_typet(std::string("float"))));

  // Call __ESBMC_dict_eq(lhs_keys, lhs_values, rhs_keys, rhs_values,
  // float_type_id)
  code_function_callt dict_eq_call;
  dict_eq_call.function() = build_symbol(*dict_eq_func);
  dict_eq_call.lhs() = build_symbol(result_var);
  dict_eq_call.arguments().push_back(lhs_keys);
  dict_eq_call.arguments().push_back(lhs_values);
  dict_eq_call.arguments().push_back(rhs_keys);
  dict_eq_call.arguments().push_back(rhs_values);
  dict_eq_call.arguments().push_back(from_integer(float_type_id, size_type()));
  dict_eq_call.type() = bool_type();
  dict_eq_call.location() = location;
  converter_.add_instruction(dict_eq_call);

  // Return result
  exprt result = build_symbol(result_var);
  result.location() = location;

  if (op == "NotEq")
  {
    exprt negated = build_not(result);
    negated.location() = location;
    return negated;
  }

  return result;
}

namespace
{
// Recognise dict-literal AST nodes whose keys and values are signed-integer
// Constants. ``dict.fromkeys([k1, k2, ...], v)`` is normalised to the same
// shape via the same recogniser before comparison.
//
// Returns nullopt for anything more complex (non-int keys, non-constant
// elements, nested dicts, ``dict.fromkeys`` with non-list-literal first arg).
// Callers must fall back to the runtime ``__ESBMC_dict_eq`` model when this
// returns nullopt.
std::optional<std::map<int64_t, int64_t>>
extract_int_int_dict_literal(const nlohmann::json &node)
{
  // Booleans intentionally fall through: ``True == 1`` in Python collapses
  // ``{True: a, 1: b}`` to a one-element dict, which ``std::map<int64_t,
  // int64_t>`` cannot model. Negative literals also fall through here because
  // they arrive as ``UnaryOp(USub, Constant(N))``, not ``Constant(-N)``.
  auto pull_int_constant =
    [](const nlohmann::json &n) -> std::optional<int64_t> {
    if (!n.contains("_type") || n["_type"] != "Constant")
      return std::nullopt;
    if (!n.contains("value") || !n["value"].is_number_integer())
      return std::nullopt;
    if (n["value"].is_boolean())
      return std::nullopt;
    return n["value"].get<int64_t>();
  };

  // Plain ``{k: v, ...}`` literal.
  if (node.contains("_type") && node["_type"] == "Dict")
  {
    if (
      !node.contains("keys") || !node.contains("values") ||
      !node["keys"].is_array() || !node["values"].is_array() ||
      node["keys"].size() != node["values"].size())
      return std::nullopt;

    std::map<int64_t, int64_t> entries;
    for (size_t i = 0; i < node["keys"].size(); ++i)
    {
      auto k = pull_int_constant(node["keys"][i]);
      auto v = pull_int_constant(node["values"][i]);
      if (!k || !v)
        return std::nullopt;
      entries[*k] = *v; // Later writes shadow earlier ones, matching Python.
    }
    return entries;
  }

  // ``dict.fromkeys([k1, k2, ...], v)`` with a list-literal first arg and a
  // constant int second arg (or default 0/None when omitted).
  if (
    node.contains("_type") && node["_type"] == "Call" &&
    node.contains("func") && node["func"].is_object() &&
    node["func"].value("_type", std::string()) == "Attribute" &&
    node["func"].value("attr", std::string()) == "fromkeys" &&
    node["func"].contains("value") &&
    node["func"]["value"].value("_type", std::string()) == "Name" &&
    node["func"]["value"].value("id", std::string()) == "dict" &&
    node.contains("args") && node["args"].is_array() && !node["args"].empty())
  {
    const auto &args = node["args"];
    const auto &keys_node = args[0];
    if (
      !keys_node.contains("_type") || keys_node["_type"] != "List" ||
      !keys_node.contains("elts") || !keys_node["elts"].is_array())
      return std::nullopt;

    // CPython defaults dict.fromkeys' fill value to None when omitted; this
    // recogniser only models int values, so refuse to fold the no-default
    // form rather than risk equating ``{k: 0}`` with ``{k: None}``.
    if (args.size() < 2)
      return std::nullopt;
    auto v = pull_int_constant(args[1]);
    if (!v)
      return std::nullopt;
    int64_t value = *v;

    std::map<int64_t, int64_t> entries;
    for (const auto &k_node : keys_node["elts"])
    {
      auto k = pull_int_constant(k_node);
      if (!k)
        return std::nullopt;
      entries[*k] = value;
    }
    return entries;
  }

  return std::nullopt;
}
} // namespace

std::optional<bool> python_dict_handler::try_constant_fold_eq(
  const nlohmann::json &lhs,
  const nlohmann::json &rhs) const
{
  auto lhs_map = extract_int_int_dict_literal(lhs);
  if (!lhs_map)
    return std::nullopt;
  auto rhs_map = extract_int_int_dict_literal(rhs);
  if (!rhs_map)
    return std::nullopt;
  return *lhs_map == *rhs_map;
}
