#include <python-frontend/python_class.h>
#include <cassert>

namespace
{
// Recursively extracts base class name (handles Name and dotted Attribute)
std::string base_name_from_ast(const json &node)
{
  if (!node.is_object())
    return {};

  const std::string type = node.value("_type", "");
  if (type == "Name")
    return node.value("id", "");

  if (type == "Attribute")
  {
    std::string left = base_name_from_ast(node.value("value", json::object()));
    std::string right = node.value("attr", "");
    if (!left.empty() && !right.empty())
      return left + "." + right;
    return right.empty() ? left : right;
  }

  return {};
}

} // namespace

void python_class::parse(const json &class_def)
{
  name_.clear();
  methods_.clear();
  attrs_.clear();
  bases_.clear();

  // 1) Class name
  if (class_def.contains("name") && class_def["name"].is_string())
    name_ = class_def["name"].get<std::string>();

  // 2) Methods and class attributes
  if (class_def.contains("body") && class_def["body"].is_array())
  {
    for (const auto &stmt : class_def["body"])
    {
      if (!stmt.is_object())
        continue;

      const std::string stmt_type = stmt.value("_type", "");

      if (stmt_type == "FunctionDef")
      {
        if (stmt.contains("name") && stmt["name"].is_string())
          methods_.insert(stmt["name"].get<std::string>());
      }
      else if (stmt_type == "Assign")
      {
        if (stmt.contains("targets") && stmt["targets"].is_array())
        {
          for (const auto &t : stmt["targets"])
          {
            if (
              t.is_object() && t.value("_type", "") == "Name" &&
              t.contains("id"))
              attrs_.insert(t["id"].get<std::string>());
          }
        }
      }
    }
  }

  // 3) Base classes
  if (class_def.contains("bases") && class_def["bases"].is_array())
  {
    for (const auto &b : class_def["bases"])
    {
      std::string base_name = base_name_from_ast(b);
      if (!base_name.empty())
        bases_.insert(base_name);
    }
  }
}
