#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

#define DUMP_OBJECT(obj) printf("%s\n", (obj).dump(2).c_str())

namespace json_utils
{
template <typename JsonType>
bool search_function_in_ast(const JsonType &node, const std::string &func_name)
{
  if (!node.is_object())
    return false;

  // Check if this is a function definition with matching name
  if (
    node.contains("_type") && node["_type"] == "FunctionDef" &&
    node.contains("name") && node["name"] == func_name)
  {
    return true;
  }

  // Recursively search all child nodes
  for (const auto &[key, value] : node.items())
  {
    if (value.is_array())
    {
      for (const auto &item : value)
      {
        if (search_function_in_ast(item, func_name))
          return true;
      }
    }
    else if (value.is_object())
    {
      if (search_function_in_ast(value, func_name))
        return true;
    }
  }
  return false;
}

template <typename JsonType>
JsonType find_class(const JsonType &ast_json, const std::string &class_name)
{
  auto it =
    std::find_if(ast_json.begin(), ast_json.end(), [&](const JsonType &obj) {
      return obj.contains("_type") && obj["_type"] == "ClassDef" &&
             obj["name"] == class_name;
    });

  return (it != ast_json.end()) ? *it : JsonType();
}

template <typename JsonType>
bool is_class(const std::string &name, const JsonType &ast_json)
{
  // Find class definition in the current json
  if (find_class(ast_json["body"], name) != JsonType())
    return true;

  // Find class definition in imported modules
  for (const auto &obj : ast_json["body"])
  {
    auto is_imported_class = [&ast_json,
                              &name](const std::string &module_name) {
      std::stringstream module_path;
      module_path << ast_json["ast_output_dir"].template get<std::string>()
                  << "/" << module_name << ".json";
      std::ifstream imported_file(module_path.str());
      if (!imported_file.is_open())
        return false;

      JsonType imported_module_json;
      imported_file >> imported_module_json;

      if (is_class(name, imported_module_json))
        return true;

      return false;
    };
    if (obj["_type"] == "ImportFrom")
      return is_imported_class(obj["module"].template get<std::string>());

    if (obj["_type"] == "Import")
    {
      for (const auto &imported : obj["names"])
      {
        if (is_imported_class(imported["name"].template get<std::string>()))
          return true;
      }
    }
  }

  return false;
}

template <typename JsonType>
bool is_module(const std::string &module_name, const JsonType &ast)
{
  std::stringstream file_path;
  file_path << ast["ast_output_dir"].template get<std::string>() << "/"
            << module_name << ".json";
  std::ifstream file(file_path.str());
  return file.is_open();
}

template <typename JsonType>
JsonType find_function(const JsonType &json, const std::string &func_name)
{
  for (const auto &elem : json)
  {
    if (elem["_type"] == "FunctionDef" && elem["name"] == func_name)
      return elem;
  }
  return JsonType();
}

template <typename JsonType>
JsonType &find_function(JsonType &json, const std::string &func_name)
{
  for (auto &elem : json)
  {
    if (elem["_type"] == "FunctionDef" && elem["name"] == func_name)
      return elem;
  }
  throw std::runtime_error("Function " + func_name + " not found\n");
}

template <typename JsonType>
const JsonType &
find_imported_function(const JsonType &ast, const std::string &func_name)
{
  for (const auto &node : ast["body"])
  {
    if (node["_type"] == "ImportFrom" || node["_type"] == "Import")
    {
      for (const auto &name : node["names"])
        if (name["name"] == func_name)
          return node;
    }
  }
  throw std::runtime_error("Function " + func_name + " not found\n");
}

template <typename JsonType>
const std::string
get_object_alias(const JsonType &ast, const std::string &obj_name)
{
  for (auto &node : ast["body"])
  {
    if (node["_type"] == "ImportFrom" || node["_type"] == "Import")
    {
      for (const auto &name : node["names"])
      {
        if (name["_type"] == "alias" && name["asname"] == obj_name)
          return name["name"];
      }
    }
  }

  std::size_t dot_pos = obj_name.rfind('.');
  if (dot_pos == std::string::npos)
    return obj_name;

  std::string prefix = obj_name.substr(0, dot_pos);
  std::string suffix = obj_name.substr(dot_pos);
  std::string resolved_prefix = get_object_alias(ast, prefix);

  if (resolved_prefix != prefix)
    return resolved_prefix + suffix;

  return obj_name;
}

template <typename JsonType>
const JsonType get_var_node(const std::string &var_name, const JsonType &block)
{
  for (auto &element : block["body"])
  {
    // Check for annotated assignment (AnnAssign)
    if (
      element["_type"] == "AnnAssign" && element.contains("target") &&
      element["target"].contains("id") && element["target"]["id"] == var_name)
      return element;

    // Check for regular assignment (Assign)
    if (
      element["_type"] == "Assign" && element.contains("targets") &&
      !element["targets"].empty() && element["targets"][0].contains("_type") &&
      element["targets"][0]["_type"] == "Name" &&
      element["targets"][0].contains("id") &&
      element["targets"][0]["id"] == var_name)
      return element;
  }

  if (block.contains("args"))
  {
    for (auto &arg : block["args"]["args"])
      if (arg.contains("arg") && arg["arg"] == var_name)
        return arg;
  }

  return JsonType();
}

template <typename JsonType>
const JsonType find_var_decl(
  const std::string &var_name,
  const std::string &function,
  const JsonType &ast)
{
  JsonType ref;

  if (!function.empty())
  {
    for (const auto &elem : ast["body"])
    {
      if (elem["_type"] == "FunctionDef" && elem["name"] == function)
        ref = get_var_node(var_name, elem);
    }
  }

  // Get variable from global scope
  if (ref.empty())
    ref = get_var_node(var_name, ast);

  return ref;
}

template <typename JsonType>
const JsonType find_return_node(const JsonType &block)
{
  for (const auto &stmt : block)
  {
    if (stmt.contains("_type") && stmt["_type"] == "Return")
      return stmt;
  }
  return JsonType();
}

} // namespace json_utils
