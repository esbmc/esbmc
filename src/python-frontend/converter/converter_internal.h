#pragma once

#include <python-frontend/json_utils.h>
#include <util/std_types.h>

#include <nlohmann/json.hpp>

#include <string>

namespace python_frontend
{

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
inline bool is_enum_class(
  const std::string &class_name,
  const nlohmann::json &ast_json)
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
