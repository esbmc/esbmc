#include <python-frontend/python_annotation/annotation_utils.h>

#include <algorithm>
#include <sstream>
#include <vector>

namespace python_annotation_utils
{

std::string get_type_from_json(const nlohmann::json &value)
{
  if (value.is_null())
    return "null";
  if (value.is_boolean())
    return "bool";
  if (value.is_number_unsigned())
    return "int";
  if (value.is_number_integer())
    return "int";
  if (value.is_number_float())
    return "float";
  if (value.is_string())
    return "str";
  if (value.is_array())
    return "array";
  if (value.is_object())
    return "object";
  return "unknown";
}

bool has_annotation(const nlohmann::json &node)
{
  return !node.empty() && node.contains("annotation") &&
         !node["annotation"].is_null();
}

bool has_return_none(const nlohmann::json &body)
{
  for (const nlohmann::json &stmt : body)
  {
    if (stmt["_type"] == "Return")
    {
      if (stmt["value"].is_null())
        return true; // bare 'return'
      if (
        stmt["value"]["_type"] == "Constant" &&
        stmt["value"]["value"].is_null())
        return true; // 'return None'
    }
    if (stmt.contains("body") && stmt["body"].is_array())
    {
      if (has_return_none(stmt["body"]))
        return true;
    }
    if (stmt.contains("orelse") && stmt["orelse"].is_array())
    {
      if (has_return_none(stmt["orelse"]))
        return true;
    }
  }
  return false;
}

std::string get_base_var_name(const nlohmann::json &node)
{
  if (node["_type"] == "Name")
    return node["id"];
  else if (node["_type"] == "Subscript")
    return get_base_var_name(node["value"]);
  else if (node["_type"] == "Attribute")
  {
    // Handle attribute access such as obj.attr[0]
    std::string base = get_base_var_name(node["value"]);
    return base + "." + node["attr"].template get<std::string>();
  }
  return "";
}

std::string invert_substrings(const std::string &input)
{
  std::vector<std::string> substrings;
  std::string token;
  std::istringstream stream(input);

  // Split the string using "." as the delimiter
  while (std::getline(stream, token, '.'))
    substrings.push_back(token);

  // Reverse the order of the substrings
  std::reverse(substrings.begin(), substrings.end());

  // Rebuild the string with "." between the reversed substrings
  std::string result;
  for (size_t i = 0; i < substrings.size(); ++i)
  {
    if (i != 0)
      result += "."; // Add the dot separator
    result += substrings[i];
  }

  return result;
}

std::string
infer_type_from_default_arg_shape(const nlohmann::json &args_node)
{
  if (!args_node.is_array() || args_node.size() < 2)
    return std::string();
  const nlohmann::json &def = args_node[1];
  const std::string def_type = def.value("_type", std::string());
  if (def_type == "List")
    return "list";
  if (def_type == "Dict")
    return "dict";
  if (def_type == "Set")
    return "set";
  if (def_type == "Tuple")
    return "tuple";
  // Scalar literal: narrow to the concrete builtin type (int/float/str/
  // bool/None) so the caller does not have to fall back to Any.
  if (def_type == "Constant" && def.contains("value"))
  {
    const std::string t = get_type_from_json(def["value"]);
    if (t == "null")
      return "None";
    if (t == "bool" || t == "int" || t == "float" || t == "str")
      return t;
  }
  return "Any";
}

} // namespace python_annotation_utils
