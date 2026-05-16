// Out-of-class member definitions for `python_annotation<Json>` that
// implement type inference. This is the bulk of the annotation pass —
// every reach into the JSON AST that asks "what type is this?" lives
// here. Definitions are kept in topological order: leaf inspectors
// first, then the dispatch entrypoints that call into them.
//
// This file is `#include`d from python_annotation.h after the class
// definition. Bodies are verbatim from the previous in-class
// definitions; only the `template <class Json>` line and the
// `python_annotation<Json>::` qualifier are new. No semantic change.

#pragma once

// ---------- leaf inspectors ----------

template <class Json>
std::string python_annotation<Json>::get_current_func_name()
{
  return current_func_name_context_;
}

template <class Json>
std::string python_annotation<Json>::get_type_from_constant(const Json &element)
{
  if (element.contains("esbmc_type_annotation"))
    return element["esbmc_type_annotation"].template get<std::string>();

  auto rhs = element["value"];

  if (rhs.is_number_integer() || rhs.is_number_unsigned())
    return "int";
  if (rhs.is_number_float())
    return "float";
  if (rhs.is_boolean())
    return "bool";
  if (rhs.is_string())
    return "str";
  if (rhs.is_null())
    return "NoneType";

  return std::string();
}

template <class Json>
std::string python_annotation<Json>::get_type_from_lhs(
  const std::string &id,
  const Json &body)
{
  // Search for LHS annotation in the current scope (e.g., while/if block)
  Json node = find_annotated_assign(id, body["body"]);

  // Fall back to the current function
  if (node.empty() && current_func != nullptr)
    node = find_annotated_assign(id, (*current_func)["body"]);

  // Fall back to variables in the global scope
  if (node.empty())
    node = find_annotated_assign(id, ast_["body"]);

  // Check if node is empty first
  if (node.empty())
    return "";

  // Check if "annotation" field exists and is not null
  if (!node.contains("annotation") || node["annotation"].is_null())
    return "";

  // Check if "annotation"."id" field exists and is not null
  if (
    !node["annotation"].contains("id") || node["annotation"]["id"].is_null())
    return "";

  // Safe to access the nested field now
  return node["annotation"]["id"];
}

template <class Json>
std::string python_annotation<Json>::get_list_subtype(const Json &list)
{
  std::string list_subtype;

  if (
    list["_type"] == "Call" && list.contains("func") &&
    list["func"].contains("attr") && list["func"]["attr"] == "array")
    return get_list_subtype(list["args"][0]);

  if (!list.contains("elts"))
    return "";

  if (!list["elts"].empty() && list["elts"][0].contains("value"))
    list_subtype = get_type_from_json(list["elts"][0]["value"]);

  for (const auto &elem : list["elts"])
    if (
      elem.contains("value") &&
      get_type_from_json(elem["value"]) != list_subtype)
      throw std::runtime_error("Multiple typed lists are not supported\n");

  return list_subtype;
}

template <class Json>
bool python_annotation<Json>::extract_type_info(
  const Json &annotation,
  std::string &base_type,
  std::string &element_type)
{
  // Handle simple Name annotations
  if (annotation.contains("id"))
  {
    std::string full_type = annotation["id"];

    // Parse generic notation such as "list[dict]"
    size_t bracket_pos = full_type.find('[');
    if (bracket_pos != std::string::npos)
    {
      base_type = full_type.substr(0, bracket_pos);
      element_type = full_type.substr(
        bracket_pos + 1, full_type.length() - bracket_pos - 2);
    }
    else
      base_type = full_type;

    return true;
  }
  return false;
}

template <class Json>
std::string python_annotation<Json>::infer_dict_value_type(const Json &var_node)
{
  if (!var_node.contains("value") || var_node["value"].is_null())
    return "Any";

  const Json &dict_init = var_node["value"];

  // Handle function calls that return dict[K, V]
  if (dict_init["_type"] == "Call" && dict_init["func"]["_type"] == "Name")
  {
    Json func_def = find_function_recursive(
      dict_init["func"]["id"].template get<std::string>(), ast_["body"]);
    if (
      !func_def.empty() && func_def.contains("returns") &&
      func_def["returns"]["_type"] == "Subscript" &&
      func_def["returns"]["value"]["id"] == "dict" &&
      func_def["returns"]["slice"]["elts"].size() >= 2)
    {
      const Json &val_type = func_def["returns"]["slice"]["elts"][1];
      if (
        val_type["_type"] == "Subscript" && val_type["value"].contains("id"))
        return val_type["value"]["id"].template get<std::string>();
    }
  }

  // Handle dict initialized from function call
  if (
    dict_init["_type"] == "Dict" && dict_init.contains("values") &&
    !dict_init["values"].empty())
  {
    const Json &first_value = dict_init["values"][0];
    std::string value_type = get_argument_type(first_value);

    if (!value_type.empty())
      return value_type;
  }

  return "Any";
}

template <class Json>
std::string python_annotation<Json>::resolve_subscript_type(
  const Json &subscript_node,
  const Json &body)
{
  const Json &base = subscript_node["value"];

  // Recursively resolve nested subscripts
  if (base["_type"] == "Subscript")
    return resolve_subscript_type(base, body);

  // Only handle Name nodes
  if (base["_type"] != "Name")
    return "Any";

  std::string var_name = base["id"];
  Json var_node =
    json_utils::find_var_decl(var_name, get_current_func_name(), ast_);

  if (
    var_node.empty() || !var_node.contains("annotation") ||
    var_node["annotation"].is_null())
    return "Any";

  std::string base_type, element_type;

  if (!extract_type_info(var_node["annotation"], base_type, element_type))
    return "Any";

  // Dict subscript: infer value type from initialization
  if (base_type == "dict")
    return infer_dict_value_type(var_node);

  // List subscript
  if (base_type == "list")
  {
    // First try to use the element_type from annotation (e.g., list[int])
    if (!element_type.empty())
      return element_type;

    // Try to infer from initialization if available
    if (var_node.contains("value") && !var_node["value"].is_null())
    {
      std::string inferred = get_list_subtype(var_node["value"]);
      if (!inferred.empty())
        return inferred;
    }

    // Last resort: return Any for unknown list element types
    return "Any";
  }

  // String subscript: str[index] returns str
  if (base_type == "str")
    return "str";

  // For other types, return the base type
  return base_type;
}

template <class Json>
bool python_annotation<Json>::is_imported_name_in_body(
  const std::string &name,
  const Json &stmts) const
{
  // Imported names inside a function body are visible to later calls in that
  // same body, even though they are not present in the module AST.
  for (const auto &stmt : stmts)
  {
    if (!stmt.contains("_type"))
      continue;

    const std::string &stype = stmt["_type"];
    if (
      (stype == "ImportFrom" || stype == "Import") && stmt.contains("names"))
    {
      for (const auto &alias : stmt["names"])
      {
        if (
          alias.contains("name") &&
          alias["name"].template get<std::string>() == name)
          return true;
        if (
          alias.contains("asname") && !alias["asname"].is_null() &&
          alias["asname"].template get<std::string>() == name)
          return true;
      }
    }
  }

  return false;
}

template <class Json>
std::string python_annotation<Json>::get_object_name(
  const Json &call,
  const std::string &prefix)
{
  if (call["value"]["_type"] == "Attribute")
  {
    std::string append = call["value"]["attr"].template get<std::string>();
    if (!prefix.empty())
      append = prefix + std::string(".") + append;

    return get_object_name(call["value"], append);
  }
  std::string obj_name("");
  if (!prefix.empty())
  {
    obj_name = prefix + std::string(".");
  }

  // Handle method calls on constant literals
  // Python allows: " ".join(l), 123.to_bytes(), etc.
  // Before this fix, accessing call["value"]["id"] would crash because
  // Constant nodes don't have an "id" field
  if (call["value"]["_type"] == "Constant")
    return get_type_from_constant(call["value"]);

  // Handle method calls on temporary instances: A().method()
  // A() is a Call node with no "id" field; return the class name so that
  // get_type_from_method() can look up the method in the class definition.
  if (
    call["value"]["_type"] == "Call" && call["value"].contains("func") &&
    call["value"]["func"]["_type"] == "Name" &&
    call["value"]["func"].contains("id"))
    return call["value"]["func"]["id"].template get<std::string>();

  // Handle chained method calls: B().g().f() where B().g() is itself a method
  // call returning an object. Recursively resolve the return type of B().g()
  // so that the outer call .f() can look up the method in the right class.
  if (
    call["value"]["_type"] == "Call" && call["value"].contains("func") &&
    call["value"]["func"]["_type"] == "Attribute")
    return get_type_from_method(call["value"]);

  // Handle normal Name values (variable references)
  if (!call["value"].contains("id"))
    return "";

  obj_name += call["value"]["id"].template get<std::string>();
  if (obj_name.find('.') != std::string::npos)
    obj_name = invert_substrings(obj_name);

  return json_utils::get_object_alias(ast_, obj_name);
}

template <class Json>
std::string python_annotation<Json>::get_string_method_return_type(
  const std::string &method) const
{
  if (
    method == "join" || method == "lower" || method == "upper" ||
    method == "strip" || method == "lstrip" || method == "rstrip" ||
    method == "format" || method == "replace")
    return "str";

  if (
    method == "startswith" || method == "endswith" || method == "isdigit" ||
    method == "isalpha" || method == "isspace" || method == "islower" ||
    method == "isupper")
    return "bool";

  if (method == "find" || method == "rfind")
    return "int";

  if (method == "split")
    return "list";

  // Keep previous behavior for unmapped string methods.
  return "str";
}

template <class Json>
std::string python_annotation<Json>::match_literal_argument(
  const Json &call_node,
  std::vector<Json> overloads) const
{
  if (overloads.empty())
    return "";

  // Try to match based on literal arguments
  if (!call_node.contains("args") || call_node["args"].empty())
    return "";

  for (const auto &overload : overloads)
  {
    if (!overload.contains("args") || !overload["args"].contains("args"))
      continue;

    const auto &params = overload["args"]["args"];
    const auto &call_args = call_node["args"];

    // Try to match first parameter (literal type check)
    if (params.size() > 0 && call_args.size() > 0)
    {
      const auto &param_annotation = params[0]["annotation"];
      const auto &call_arg = call_args[0];

      // Check for Literal["foo"] pattern
      if (
        param_annotation["_type"] == "Subscript" &&
        param_annotation["value"]["id"] == "Literal" &&
        param_annotation["slice"]["_type"] == "Constant")
      {
        std::string literal_value =
          param_annotation["slice"]["value"].template get<std::string>();

        // Check if call argument matches
        if (
          call_arg["_type"] == "Constant" && call_arg["value"].is_string() &&
          call_arg["value"].template get<std::string>() == literal_value)
        {
          // Found matching overload, return its type
          if (
            overload.contains("returns") && !overload["returns"].is_null() &&
            overload["returns"].contains("id"))
          {
            return overload["returns"]["id"];
          }
        }
      }
    }
  }

  return "";
}

// Find the best matching overload
template <class Json>
std::string python_annotation<Json>::resolve_overload_return_type(
  const std::string &func_name,
  const Json &call_node) const
{
  std::vector<Json> overloads;

  // Find all overload definitions
  for (const Json &elem : ast_["body"])
  {
    if (
      elem["_type"] == "FunctionDef" && elem["name"] == func_name &&
      json_utils::has_overload_decorator(elem))
      overloads.push_back(elem);
  }

  return match_literal_argument(call_node, overloads);
}

template <class Json>
std::string
python_annotation<Json>::resolve_object_class_name(const std::string &obj)
{
  auto read_name_id = [](const Json &annotation) -> std::string {
    if (
      annotation.is_object() && !annotation.is_null() &&
      annotation.contains("_type") && annotation["_type"] == "Name" &&
      annotation.contains("id") && annotation["id"].is_string())
      return annotation["id"].template get<std::string>();
    return "";
  };

  if (
    current_func != nullptr && (*current_func).contains("args") &&
    (*current_func)["args"].contains("args"))
  {
    Json param = find_annotated_assign(obj, (*current_func)["args"]["args"]);
    if (param.contains("annotation"))
    {
      std::string id = read_name_id(param["annotation"]);
      if (!id.empty())
        return id;
    }
  }

  Json var = json_utils::find_var_decl(obj, get_current_func_name(), ast_);
  if (var.empty())
    return "";

  if (var.contains("annotation"))
  {
    std::string id = read_name_id(var["annotation"]);
    if (!id.empty())
      return id;
  }

  if (
    var.contains("value") && var["value"].is_object() &&
    var["value"].contains("_type") && var["value"]["_type"] == "Call" &&
    var["value"].contains("func") && var["value"]["func"].is_object() &&
    var["value"]["func"].contains("_type") &&
    var["value"]["func"]["_type"] == "Name" &&
    var["value"]["func"].contains("id") &&
    var["value"]["func"]["id"].is_string())
    return var["value"]["func"]["id"].template get<std::string>();

  return "";
}

template <class Json>
std::string python_annotation<Json>::infer_unpacked_element_type(
  const Json &rhs,
  size_t index)
{
  if (!rhs.is_object() || !rhs.contains("_type"))
    return "Any";

  const std::string &kind = rhs["_type"];

  if (
    (kind == "Tuple" || kind == "List") && rhs.contains("elts") &&
    index < rhs["elts"].size())
  {
    std::string t = get_argument_type(rhs["elts"][index]);
    return t.empty() ? "Any" : t;
  }

  if (
    kind == "Attribute" && rhs.contains("value") &&
    rhs["value"].is_object() && rhs["value"]["_type"] == "Name" &&
    rhs.contains("attr"))
  {
    std::string cls = resolve_object_class_name(
      rhs["value"]["id"].template get<std::string>());
    if (cls.empty())
      return "Any";
    Json attr_rhs = python_annotation_parser::find_self_attr_init_rhs(
      cls, rhs["attr"].template get<std::string>(), ast_["body"]);
    if (attr_rhs.empty())
      return "Any";
    return infer_unpacked_element_type(attr_rhs, index);
  }

  return "Any";
}
