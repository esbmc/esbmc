#pragma once

#include <python-frontend/json_utils.h>
#include <python-frontend/type/type_utils.h>
#include <util/message/message.h>
#include <util/irep/std_types.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <optional>
#include <string>
#include <unordered_map>

namespace python_frontend
{
// Python AST statement-id (e.g. "If", "Return") -> internal StatementType.
inline StatementType get_statement_type(const nlohmann::json &element)
{
  static const std::unordered_map<std::string, StatementType> statement_map = {
    {"AnnAssign", StatementType::VARIABLE_ASSIGN},
    {"Assign", StatementType::VARIABLE_ASSIGN},
    {"FunctionDef", StatementType::FUNC_DEFINITION},
    {"If", StatementType::IF_STATEMENT},
    {"AugAssign", StatementType::COMPOUND_ASSIGN},
    {"While", StatementType::WHILE_STATEMENT},
    {"For", StatementType::FOR_STATEMENT},
    {"Expr", StatementType::EXPR},
    {"Return", StatementType::RETURN},
    {"Assert", StatementType::ASSERT},
    {"ClassDef", StatementType::CLASS_DEFINITION},
    {"Pass", StatementType::PASS},
    {"Break", StatementType::BREAK},
    {"Continue", StatementType::CONTINUE},
    {"ImportFrom", StatementType::IMPORT},
    {"Import", StatementType::IMPORT},
    {"Raise", StatementType::RAISE},
    {"Global", StatementType::GLOBAL},
    {"Try", StatementType::TRY},
    {"ExceptHandler", StatementType::EXCEPTHANDLER},
    {"Delete", StatementType::DELETE_STATEMENT}};

  if (!element.contains("_type"))
    return StatementType::UNKNOWN;

  auto it = statement_map.find(element["_type"]);
  return (it != statement_map.end()) ? it->second : StatementType::UNKNOWN;
}

// Operator name (Python AST id, e.g. "Add", "Lt") -> ESBMC operator id.
inline const std::unordered_map<std::string, std::string> &operator_map()
{
  static const std::unordered_map<std::string, std::string> m = {
    {"add", "+"},         {"sub", "-"},         {"subtract", "-"},
    {"mult", "*"},        {"multiply", "*"},    {"dot", "*"},
    {"div", "/"},         {"divide", "/"},      {"mod", "mod"},
    {"bitor", "bitor"},   {"floordiv", "/"},    {"bitand", "bitand"},
    {"bitxor", "bitxor"}, {"invert", "bitnot"}, {"lshift", "shl"},
    {"rshift", "ashr"},   {"usub", "unary-"},   {"eq", "="},
    {"lt", "<"},          {"lte", "<="},        {"noteq", "notequal"},
    {"gt", ">"},          {"gte", ">="},        {"and", "and"},
    {"or", "or"},         {"not", "not"},       {"uadd", "unary+"},
    {"is", "="},          {"isnot", "not"},     {"in", "="}};
  return m;
}

// Map a Python operator name to its ESBMC representation. Uses IEEE-specific
// operators when the type is floating-point.
inline std::string map_operator(const std::string &op, const typet &type)
{
  // Convert the operator to lowercase to allow case-insensitive comparison.
  std::string lower_op = op;
  std::transform(
    lower_op.begin(), lower_op.end(), lower_op.begin(), [](unsigned char c) {
      return std::tolower(c);
    });

  // If the type is floating-point, use IEEE-specific operators.
  if (type.is_floatbv())
  {
    static const std::unordered_map<std::string, std::string> float_ops = {
      {"add", "ieee_add"},
      {"sub", "ieee_sub"},
      {"subtract", "ieee_sub"},
      {"mult", "ieee_mul"},
      {"dot", "ieee_mul"},
      {"multiply", "ieee_mul"},
      {"div", "ieee_div"},
      {"divide", "ieee_div"}};

    auto float_it = float_ops.find(lower_op);
    if (float_it != float_ops.end())
      return float_it->second;
  }

  // Look up the operator in the general operator map (for non-floating-point
  // types).
  const auto &m = operator_map();
  auto it = m.find(lower_op);
  if (it != m.end())
    return it->second;

  log_warning("Unknown operator: {}", op);
  return {};
}

// Build a struct member component tagged for its owning class.
inline struct_typet::componentt build_component(
  const std::string &class_name,
  const std::string &comp_name,
  const typet &type)
{
  struct_typet::componentt component(comp_name, comp_name, type);

  // Add metadata used internally by ESBMC for member-to-class tagging.
  // The key "#member_name" is used by the type system; the value
  // "tag-<class_name>" helps associate this member with its parent class.
  type_utils::set_member_name(component.type(), "tag-" + class_name);

  // Set the member visibility to public by default.
  component.set_access("public");

  return component;
}

// Returns true if the named class inherits from Python's Enum.
inline bool
is_enum_class(const std::string &class_name, const nlohmann::json &ast_json)
{
  const nlohmann::json class_node =
    json_utils::find_class(ast_json["body"], class_name);
  if (
    class_node.empty() || !class_node.contains("bases") ||
    !class_node["bases"].is_array())
    return false;
  for (const auto &base : class_node["bases"])
    if (base.is_object() && base.contains("id"))
    {
      // Resolve any import alias (e.g. "from enum import Enum as E" → "Enum")
      const std::string resolved =
        json_utils::get_object_alias(ast_json, base["id"].get<std::string>());
      if (resolved == "Enum")
        return true;
    }
  return false;
}

// An integer literal, possibly negated (`3`, `-3`).
inline bool is_literal_int_node(const nlohmann::json &node)
{
  if (!node.is_object())
    return false;
  if (
    node.value("_type", "") == "Constant" && node.contains("value") &&
    node["value"].is_number_integer())
    return true;

  return node.value("_type", "") == "UnaryOp" && node.contains("op") &&
         node["op"].value("_type", "") == "USub" && node.contains("operand") &&
         node["operand"].value("_type", "") == "Constant" &&
         node["operand"].contains("value") &&
         node["operand"]["value"].is_number_integer();
}

inline std::optional<long long> literal_int_value(const nlohmann::json &node)
{
  if (!is_literal_int_node(node))
    return std::nullopt;

  if (node.value("_type", "") == "Constant")
    return node["value"].get<long long>();

  return -node["operand"]["value"].get<long long>();
}

// The literal step of a Slice node (1 when omitted); nullopt when it is not
// a literal or is zero.
inline std::optional<long long> literal_slice_step(const nlohmann::json &slice)
{
  if (!slice.contains("step") || slice["step"].is_null())
    return 1;

  std::optional<long long> step = literal_int_value(slice["step"]);
  if (!step || *step == 0)
    return std::nullopt;
  return step;
}

// One literal bound of a Slice node, resolved with Python's rules against an
// axis of `axis_len` elements; nullopt when the bound is not a literal.
inline std::optional<long long> literal_slice_bound(
  const nlohmann::json &slice,
  const char *name,
  const long long axis_len,
  const long long step)
{
  const bool lower = std::string(name) == "lower";
  if (!slice.contains(name) || slice[name].is_null())
  {
    if (step > 0)
      return lower ? 0 : axis_len;
    return lower ? axis_len - 1 : -1;
  }

  std::optional<long long> value = literal_int_value(slice[name]);
  if (!value)
    return std::nullopt;

  const long long resolved = *value < 0 ? *value + axis_len : *value;
  if (step > 0)
    return std::max(0LL, std::min(resolved, axis_len));
  return std::max(-1LL, std::min(resolved, axis_len - 1));
}

// Number of elements a Slice with literal bounds and step selects from an
// axis of `dim` elements; nullopt when any of them is not a literal.
inline std::optional<std::size_t>
literal_slice_length(const std::size_t dim, const nlohmann::json &slice)
{
  const std::optional<long long> step = literal_slice_step(slice);
  if (!step)
    return std::nullopt;

  const long long axis_len = static_cast<long long>(dim);
  const std::optional<long long> start =
    literal_slice_bound(slice, "lower", axis_len, *step);
  const std::optional<long long> stop =
    literal_slice_bound(slice, "upper", axis_len, *step);
  if (!start || !stop)
    return std::nullopt;

  const long long span = *step > 0 ? *stop - *start : *start - *stop;
  const long long stride = *step > 0 ? *step : -*step;
  return span <= 0 ? 0 : static_cast<std::size_t>((span - 1) / stride + 1);
}

} // namespace python_frontend
