#pragma once

#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <functional>

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
  auto is_main_guard = [&](const JsonType &node) -> bool {
    if (
      !node.contains("_type") || node["_type"] != "If" ||
      !node.contains("test"))
      return false;

    const auto &test = node["test"];
    if (
      !test.contains("_type") || test["_type"] != "Compare" ||
      !test.contains("left") || !test.contains("comparators") ||
      !test.contains("ops") || !test["comparators"].is_array() ||
      test["comparators"].empty())
      return false;

    const auto &left = test["left"];
    if (
      !left.contains("_type") || left["_type"] != "Name" ||
      !left.contains("id") || left["id"] != "__name__")
      return false;

    const auto &ops = test["ops"];
    if (!ops.is_array() || ops.empty())
      return false;

    bool eq_op = false;
    if (ops[0].is_object() && ops[0].contains("_type"))
      eq_op = ops[0]["_type"] == "Eq";
    else if (ops[0].is_string())
      eq_op = ops[0] == "Eq";

    if (!eq_op)
      return false;

    const auto &comp = test["comparators"][0];
    if (
      !comp.contains("_type") || comp["_type"] != "Constant" ||
      !comp.contains("value") || !comp["value"].is_string())
      return false;

    return comp["value"] == "__main__";
  };

  for (const auto &element : block["body"])
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

    auto search_if_body = [&](const JsonType &if_node) -> JsonType {
      JsonType found = get_var_node(var_name, if_node);
      if (!found.empty())
        return found;

      if (
        if_node.contains("orelse") && if_node["orelse"].is_array() &&
        !if_node["orelse"].empty())
      {
        JsonType orelse_block;
        orelse_block["body"] = if_node["orelse"];
        return get_var_node(var_name, orelse_block);
      }

      return JsonType();
    };

    if (
      element.contains("_type") && element["_type"] == "If" &&
      element.contains("body") && element["body"].is_array())
    {
      if (is_main_guard(element))
      {
        JsonType found = search_if_body(element);
        if (!found.empty())
          return found;
      }
      else
      {
        JsonType found = search_if_body(element);
        if (!found.empty())
          return found;
      }
    }
  }

  if (block.contains("args"))
  {
    for (auto &arg : block["args"]["args"])
      if (arg.contains("arg") && arg["arg"] == var_name)
        return arg;
  }

  return JsonType();
}

// Split function path "foo@F@bar@F@baz" -> ["foo", "bar", "baz"]
inline std::vector<std::string> split_function_path(const std::string &function)
{
  std::vector<std::string> path;
  size_t start = 0;
  size_t pos = function.find("@F@");

  while (pos != std::string::npos)
  {
    path.push_back(function.substr(start, pos - start));
    start = pos + 3; // Skip "@F@"
    pos = function.find("@F@", start);
  }

  if (start < function.length())
    path.push_back(function.substr(start));

  return path;
}

// Find a function in AST by hierarchical path
// Example: ["foo", "bar"] finds nested function bar() inside foo()
template <typename JsonType>
JsonType
find_function_by_path(const JsonType &ast, const std::vector<std::string> &path)
{
  if (path.empty())
    return JsonType();

  std::function<JsonType(const JsonType &, size_t)> search_recursive;
  search_recursive =
    [&](const JsonType &parent_body, size_t depth) -> JsonType {
    if (depth >= path.size())
      return JsonType();

    const std::string &target_name = path[depth];

    for (const auto &elem : parent_body)
    {
      if (elem["_type"] == "FunctionDef" && elem["name"] == target_name)
      {
        // Found at this level
        if (depth == path.size() - 1)
          return elem; // This is the target function

        // Need to go deeper into nested functions
        if (elem.contains("body") && elem["body"].is_array())
          return search_recursive(elem["body"], depth + 1);
      }
    }
    return JsonType();
  };

  // First, try to find in top-level functions
  JsonType result = search_recursive(ast["body"], 0);
  if (!result.empty())
    return result;

  // If not found, search inside class methods
  for (const auto &elem : ast["body"])
  {
    if (elem["_type"] == "ClassDef" && elem.contains("body"))
    {
      result = search_recursive(elem["body"], 0);
      if (!result.empty())
        return result;
    }
  }

  return JsonType(); // Not found
}

// Find variable in a specific function node (searches params + body)
template <typename JsonType>
JsonType
find_var_in_function(const std::string &var_name, const JsonType &func_node)
{
  if (func_node.empty())
    return JsonType();

  // First, search in function parameters
  if (func_node.contains("args") && func_node["args"].contains("args"))
  {
    for (const auto &arg : func_node["args"]["args"])
    {
      if (arg.contains("arg") && arg["arg"] == var_name)
        return arg;
    }
  }

  // Search in keyword-only arguments (after *)
  if (func_node.contains("args") && func_node["args"].contains("kwonlyargs"))
  {
    for (const auto &arg : func_node["args"]["kwonlyargs"])
    {
      if (arg.contains("arg") && arg["arg"] == var_name)
        return arg;
    }
  }

  // Then search in function body
  return get_var_node(var_name, func_node);
}

template <typename JsonType>
const JsonType find_var_decl(
  const std::string &var_name,
  const std::string &function,
  const JsonType &ast)
{
  // If no function context, search in global scope
  if (function.empty())
    return get_var_node(var_name, ast);

  // Parse function path (e.g., "foo@F@bar" -> ["foo", "bar"])
  std::vector<std::string> function_path = split_function_path(function);

  // Search from innermost to outermost scope (closure semantics)
  // Example: for "foo@F@bar", search first in bar(), then in foo()
  for (int scope_level = static_cast<int>(function_path.size()) - 1;
       scope_level >= 0;
       --scope_level)
  {
    // Build partial path up to current scope level
    std::vector<std::string> partial_path(
      function_path.begin(), function_path.begin() + scope_level + 1);

    // Find the function node at this scope level
    JsonType func_node = find_function_by_path(ast, partial_path);

    if (!func_node.empty())
    {
      // Search for variable in this function's scope
      JsonType var = find_var_in_function(var_name, func_node);
      if (!var.empty())
        return var; // Found in this scope
    }
  }

  // Fallback: search in global scope
  return get_var_node(var_name, ast);
}

template <typename JsonType>
const JsonType get_var_value(
  const std::string &var_name,
  const std::string &function,
  const JsonType &ast)
{
  JsonType value = find_var_decl(var_name, function, ast);
  while (!value.empty() && value["_type"] != "arg" &&
         value["value"]["_type"] == "Name")
  {
    value = find_var_decl(value["value"]["id"], function, ast);
  }
  return value;
}

template <typename JsonType>
bool extract_constant_integer(
  const JsonType &node,
  const std::string &function,
  const JsonType &ast,
  long long &value)
{
  if (
    node.contains("_type") && node["_type"] == "Constant" &&
    node.contains("value") && node["value"].is_number_integer())
  {
    value = node["value"].template get<long long>();
    return true;
  }

  if (
    node.contains("_type") && node["_type"] == "UnaryOp" &&
    node.contains("operand") && node["operand"].contains("value") &&
    node["operand"]["value"].is_number_integer() && node.contains("op"))
  {
    const auto &op = node["op"];
    const bool is_usub =
      (op.is_object() && op.contains("_type") && op["_type"] == "USub") ||
      (op.is_string() && op == "USub");
    const bool is_uadd =
      (op.is_object() && op.contains("_type") && op["_type"] == "UAdd") ||
      (op.is_string() && op == "UAdd");

    if (is_usub || is_uadd)
    {
      value = node["operand"]["value"].template get<long long>();
      if (is_usub)
        value = -value;
      return true;
    }
  }

  if (node.contains("_type") && node["_type"] == "Name" && node.contains("id"))
  {
    const std::string var_name = node["id"].template get<std::string>();
    JsonType var_value = get_var_value(var_name, function, ast);

    if (
      !var_value.empty() && var_value.contains("value") &&
      var_value["value"].contains("_type") &&
      var_value["value"]["_type"] == "Constant" &&
      var_value["value"].contains("value") &&
      var_value["value"]["value"].is_number_integer())
    {
      value = var_value["value"]["value"].template get<long long>();
      return true;
    }
  }

  return false;
}

template <typename JsonType>
const JsonType get_list_element(const JsonType &list_value, int pos)
{
  // Handle direct List node
  if (
    list_value["_type"] == "List" && list_value.contains("elts") &&
    !list_value["elts"].empty())
  {
    return list_value["elts"][pos];
  }

  // Handle BinOp (e.g., list concatenation or repetition)
  if (list_value["_type"] == "BinOp")
  {
    if (list_value["left"]["_type"] == "List")
      return list_value["left"]["elts"][pos];
    if (list_value["right"]["_type"] == "List")
      return list_value["right"]["elts"][pos];
  }

  // Handle Subscript (e.g., d['a'] where d is a dict containing lists)
  // Return empty JSON: caller should use type annotations instead
  if (list_value["_type"] == "Subscript")
    return JsonType();

  // Handle Name reference (variable that holds a list)
  // Return empty JSON: caller should resolve the variable
  if (list_value["_type"] == "Name")
    return JsonType();

  return JsonType();
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

/// Extract the variable name from a symbol identifier
/// Examples:
///   "py:test.py@l" -> "l"
///   "py:test.py@F@foo@x" -> "x"
///   "py:test.py@C@MyClass@F@method@var" -> "var"
inline std::string extract_var_name_from_symbol_id(const std::string &symbol_id)
{
  size_t last_at = symbol_id.find_last_of('@');
  return (last_at != std::string::npos) ? symbol_id.substr(last_at + 1)
                                        : symbol_id;
}

template <typename JsonType>
bool has_overload_decorator(const JsonType &func_node)
{
  // Check for @overload decorators
  if (!func_node.contains("decorator_list"))
    return false;

  for (const auto &decorator : func_node["decorator_list"])
  {
    if (decorator["_type"] == "Name" && decorator["id"] == "overload")
      return true;
  }
  return false;
}

} // namespace json_utils
