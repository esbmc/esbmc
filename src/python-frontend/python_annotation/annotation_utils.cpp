#include <python-frontend/python_annotation/annotation_utils.h>

#include <algorithm>
#include <functional>
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

bool has_return_value(const nlohmann::json &body)
{
  for (const nlohmann::json &stmt : body)
  {
    if (
      stmt["_type"] == "Return" && !stmt["value"].is_null() &&
      !(stmt["value"]["_type"] == "Constant" &&
        stmt["value"]["value"].is_null()))
      return true;
    for (const char *key : {"body", "orelse"})
      if (
        stmt.contains(key) && stmt[key].is_array() &&
        has_return_value(stmt[key]))
        return true;
  }
  return false;
}

// Whether `node`, outside nested loops and scopes, has a `break`.
static bool breaks(const nlohmann::json &node)
{
  if (node.is_object())
  {
    const std::string kind = node.value("_type", "");
    if (kind == "Break")
      return true;
    if (
      kind == "For" || kind == "AsyncFor" || kind == "While" ||
      kind == "FunctionDef" || kind == "AsyncFunctionDef" ||
      kind == "ClassDef" || kind == "Lambda")
      return false;
  }
  if (node.is_structured())
    for (const nlohmann::json &child : node)
      if (breaks(child))
        return true;
  return false;
}

// Whether no path through `block` reaches its end.
static bool never_falls_off(const nlohmann::json &block)
{
  if (!block.is_array() || block.empty())
    return false;
  const nlohmann::json &last = block.back();
  const std::string kind = last.value("_type", "");
  if (kind == "Return" || kind == "Raise")
    return true;
  if (kind == "If")
    return never_falls_off(last["body"]) && never_falls_off(last["orelse"]);
  if (kind == "While")
    return last["test"].value("_type", "") == "Constant" &&
           last["test"].value("value", nlohmann::json()) == true &&
           !breaks(last["body"]);
  if (kind != "Try")
    return false;
  const nlohmann::json &handlers = last["handlers"];
  return never_falls_off(last["finalbody"]) ||
         ((never_falls_off(last["body"]) || never_falls_off(last["orelse"])) &&
          std::all_of(
            handlers.begin(), handlers.end(), [](const nlohmann::json &h) {
              return never_falls_off(h["body"]);
            }));
}

// Whether `node`, outside nested scopes, returns a value other than None and
// does not yield.
static bool returns_value_without_yield(const nlohmann::json &node)
{
  bool value_return = false;
  bool yields = false;
  std::function<void(const nlohmann::json &)> scan =
    [&](const nlohmann::json &n) {
      if (n.is_object())
      {
        const std::string kind = n.value("_type", "");
        if (
          kind == "FunctionDef" || kind == "AsyncFunctionDef" ||
          kind == "ClassDef" || kind == "Lambda")
          return;
        yields |= kind == "Yield" || kind == "YieldFrom";
        value_return |=
          kind == "Return" && n["value"].is_object() &&
          !(n["value"].value("_type", "") == "Constant" &&
            n["value"]["value"].is_null() && !n["value"].contains("_bigint"));
      }
      if (n.is_structured())
        for (const nlohmann::json &child : n)
          scan(child);
    };
  scan(node);
  return value_return && !yields;
}

// `node[key]` when it is a number, else `fallback`. A statement the
// preprocessor synthesised can carry a null location field.
static int number_or(const nlohmann::json &node, const char *key, int fallback)
{
  const auto it = node.find(key);
  return it != node.end() && it->is_number() ? it->get<int>() : fallback;
}

void return_none_on_fall_off(nlohmann::json &node)
{
  if (!node.is_structured())
    return;
  for (nlohmann::json &child : node)
    return_none_on_fall_off(child);
  if (
    !node.is_object() || node.value("_type", "") != "FunctionDef" ||
    !node.value("returns", nlohmann::json()).is_null() ||
    never_falls_off(node["body"]) || !returns_value_without_yield(node["body"]))
    return;
  nlohmann::json &body = node["body"];
  const int line =
    number_or(body.back(), "end_lineno", number_or(node, "lineno", 0));
  const int col = number_or(body.back(), "col_offset", 0);
  const nlohmann::json at = {
    {"lineno", line},
    {"col_offset", col},
    {"end_lineno", line},
    {"end_col_offset", col}};
  nlohmann::json none = at;
  none["_type"] = "Constant";
  none["value"] = nullptr;
  nlohmann::json ret = at;
  ret["_type"] = "Return";
  ret["value"] = none;
  body.push_back(ret);
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

std::string infer_type_from_default_arg_shape(const nlohmann::json &args_node)
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
