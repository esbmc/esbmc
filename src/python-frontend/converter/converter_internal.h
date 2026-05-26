#pragma once

#include <python-frontend/json_utils.h>
#include <python-frontend/type_utils.h>
#include <util/message.h>
#include <util/std_types.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
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

  // Look up the operator in the general operator map (for non-floating-point types).
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
  component.type().set("#member_name", "tag-" + class_name);

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

} // namespace python_frontend
