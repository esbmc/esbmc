#include <python-frontend/python_annotation/annotation_parser.h>

#include <python-frontend/json_utils.h>

#include <cassert>

namespace python_annotation_parser
{
nlohmann::json
find_lambda_in_body(const std::string &func_name, const nlohmann::json &body)
{
  for (const nlohmann::json &elem : body)
  {
    if (
      elem["_type"] == "Assign" && !elem["targets"].empty() &&
      elem["targets"][0].contains("id") &&
      elem["targets"][0]["id"] == func_name && elem.contains("value") &&
      elem["value"]["_type"] == "Lambda")
    {
      return elem;
    }
  }
  return nlohmann::json();
}

nlohmann::json find_function_recursive(
  const std::string &func_name,
  const nlohmann::json &body)
{
  for (const nlohmann::json &elem : body)
  {
    // Found the function at this level
    if (elem["_type"] == "FunctionDef" && elem["name"] == func_name)
    {
      return elem;
    }

    // Recursively search in nested function bodies
    if (elem["_type"] == "FunctionDef" && elem.contains("body"))
    {
      nlohmann::json nested = find_function_recursive(func_name, elem["body"]);
      if (!nested.empty())
        return nested;
    }

    // Also search in control flow blocks
    if (elem.contains("body") && elem["body"].is_array())
    {
      nlohmann::json nested = find_function_recursive(func_name, elem["body"]);
      if (!nested.empty())
        return nested;
    }

    if (elem.contains("orelse") && elem["orelse"].is_array())
    {
      nlohmann::json nested =
        find_function_recursive(func_name, elem["orelse"]);
      if (!nested.empty())
        return nested;
    }
  }

  return nlohmann::json(); // Not found
}

nlohmann::json find_self_attr_init_rhs(
  const std::string &cls,
  const std::string &attr,
  const nlohmann::json &ast_body)
{
  // @p ast_body is the top-level AST `body` array (i.e. ast_["body"]).
  // `json_utils::find_class` iterates it expecting class-shaped nodes.
  assert(ast_body.is_array());
  nlohmann::json class_node = json_utils::find_class(ast_body, cls);
  if (class_node.empty() || !class_node.contains("body"))
    return nlohmann::json();

  for (const nlohmann::json &member : class_node["body"])
  {
    if (member["_type"] != "FunctionDef" || member["name"] != "__init__")
      continue;
    for (const nlohmann::json &stmt : member["body"])
    {
      if (!stmt.contains("_type") || !stmt.contains("value"))
        continue;
      const nlohmann::json *target = nullptr;
      if (stmt["_type"] == "AnnAssign" && stmt.contains("target"))
        target = &stmt["target"];
      else if (
        stmt["_type"] == "Assign" && stmt.contains("targets") &&
        stmt["targets"].is_array() && !stmt["targets"].empty())
        target = &stmt["targets"][0];
      if (
        target == nullptr || !target->is_object() ||
        !target->contains("_type") || (*target)["_type"] != "Attribute" ||
        !target->contains("value") || !(*target)["value"].is_object() ||
        !(*target)["value"].contains("_type") ||
        (*target)["value"]["_type"] != "Name" ||
        !(*target)["value"].contains("id") ||
        (*target)["value"]["id"] != "self" || !target->contains("attr") ||
        (*target)["attr"] != attr)
        continue;
      return stmt["value"];
    }
  }
  return nlohmann::json();
}

} // namespace python_annotation_parser
