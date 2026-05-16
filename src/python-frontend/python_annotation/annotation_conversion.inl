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

// ---------- dispatchers ----------

template <class Json>
std::string python_annotation<Json>::get_argument_type(const Json &arg)
{
  if (arg["_type"] == "Constant")
    return get_type_from_constant(arg);
  else if (arg["_type"] == "Subscript")
  {
    // Handle subscripts like tokens[0] and slices like tokens[:1].
    const auto &val = arg["value"];
    if (val["_type"] == "Name")
    {
      std::string var_name = val["id"].template get<std::string>();
      Json var_node = find_var_node_for_inference(var_name);

      if (has_annotation(var_node))
      {
        const auto &annot = var_node["annotation"];
        // list[T] -> return T
        if (
          annot.contains("_type") && annot["_type"] == "Subscript" &&
          annot.contains("value") && annot["value"].contains("id") &&
          annot["value"]["id"] == "list")
        {
          if (
            arg.contains("slice") && arg["slice"].contains("_type") &&
            arg["slice"]["_type"] == "Slice")
          {
            if (
              annot.contains("slice") && annot["slice"].contains("id") &&
              annot["slice"]["id"].is_string())
              return "list[" +
                     annot["slice"]["id"].template get<std::string>() + "]";
            return "list";
          }

          if (annot.contains("slice"))
          {
            const auto &slice = annot["slice"];
            if (slice.contains("id"))
              return slice["id"];
            else if (
              slice.contains("_type") && slice["_type"] == "Name" &&
              slice.contains("id"))
              return slice["id"];
          }
          return "Any";
        }
        // Simple name annotation (e.g., list without subtype)
        if (annot.contains("id") && annot["id"] == "list")
          return "Any";
      }
    }
    return "";
  }
  else if (arg["_type"] == "Slice")
    return "slice";
  else if (arg["_type"] == "UnaryOp")
  {
    // Handle unary operations like -5, +3, not True
    if (arg.contains("operand"))
      return get_argument_type(arg["operand"]);
  }
  else if (arg["_type"] == "Name")
  {
    const std::string &var_name = arg["id"];

    // Try to find the type of the variable in current scope
    Json var_node = find_var_node_for_inference(var_name);

    // Extract type from the found variable node
    if (has_annotation(var_node))
    {
      // Handle arg nodes (function parameters)
      if (var_node["_type"] == "arg")
      {
        return json_utils::get_annotation_type_name(var_node["annotation"]);
      }
      // Handle generic type annotations like list[int] (Subscript nodes)
      else if (
        var_node["annotation"].contains("_type") &&
        var_node["annotation"]["_type"] == "Subscript" &&
        var_node["annotation"].contains("value") &&
        var_node["annotation"]["value"].contains("id"))
      {
        return json_utils::get_annotation_type_name(var_node["annotation"]);
      }
      // Handle simple type annotations like int, str (Name nodes)
      else if (var_node["annotation"].contains("id"))
      {
        std::string base_type =
          var_node["annotation"]["id"].template get<std::string>();
        // If annotation is just "list"/"dict"/"set" without element type, try to infer from value
        if (
          (base_type == "list" || base_type == "dict" ||
           base_type == "set") &&
          var_node.contains("value") && !var_node["value"].is_null())
        {
          if (base_type == "list" && var_node["value"]["_type"] == "List")
          {
            std::string full_type =
              get_list_type_from_literal(var_node["value"]);
            if (!full_type.empty())
              return full_type;
          }
        }
        return base_type;
      }
    }

    if (
      !var_node.empty() && var_node.contains("value") &&
      !var_node["value"].is_null())
    {
      std::string inferred_type = get_argument_type(var_node["value"]);
      if (!inferred_type.empty())
        return inferred_type;
    }
  }
  else if (arg["_type"] == "BinOp")
  {
    // For binary operations, try to infer from operands
    Json dummy_stmt = {{"value", arg}};
    return get_type_from_binary_expr(dummy_stmt, ast_);
  }
  else if (arg["_type"] == "List")
  {
    // Handle list literals like [-5, 1, 3, 5, 7, 10]
    return get_list_type_from_literal(arg);
  }
  else if (arg["_type"] == "Dict")
    return "dict";
  else if (arg["_type"] == "Set")
    return "set";
  else if (arg["_type"] == "Tuple")
    return "tuple";
  else if (arg["_type"] == "BoolOp")
  {
    if (arg.contains("values") && arg["values"].is_array())
    {
      for (const auto &val : arg["values"])
      {
        if (
          val.contains("_type") && val["_type"] == "Call" &&
          val.contains("func") && val["func"].contains("_type") &&
          val["func"]["_type"] == "Name" && val["func"].contains("id"))
        {
          auto it = builtin_functions().find(val["func"]["id"]);
          if (it != builtin_functions().end())
            return it->second;
        }
      }
    }
    return "bool";
  }
  else if (arg["_type"] == "Call")
  {
    // Handle function calls like abs(a - b), len(list), etc.
    if (arg["func"]["_type"] == "Name")
    {
      const std::string &func_name = arg["func"]["id"];

      // Class constructor call: A() produces an instance of A
      // Resolve possible import aliases before checking if this is a class
      const std::string class_name =
        json_utils::get_object_alias(ast_, func_name);
      if (json_utils::is_class(class_name, ast_))
        return class_name;

      // Check built-in functions first
      auto it = builtin_functions().find(func_name);
      if (it != builtin_functions().end())
        return it->second;

      // For user-defined functions, try to get return type
      try
      {
        return get_function_return_type(func_name, ast_);
      }
      catch (std::runtime_error &)
      {
        // Function not found, return empty
        return "";
      }
    }
  }

  return ""; // Cannot determine type
}

// Method to get the full type of a list literal
template <class Json>
std::string
python_annotation<Json>::get_list_type_from_literal(const Json &list_arg)
{
  if (!list_arg.contains("elts") || list_arg["elts"].empty())
  {
    log_warning(
      "Empty or malformed list literal detected. Using 'list[int]' as "
      "default ({}:{})",
      python_filename_,
      current_line_);
    return "list"; // Default fallback
  }

  // Get the type of the first element
  std::string element_type = get_argument_type(list_arg["elts"][0]);

  if (element_type.empty())
  {
    log_warning(
      "Could not determine list element type. Using 'list[int]' as fallback "
      "({}:{})",
      python_filename_,
      current_line_);
    return "list[int]"; // Fallback for numeric contexts
  }

  // Check if all elements have the same type
  for (size_t i = 1; i < list_arg["elts"].size(); ++i)
  {
    std::string current_type = get_argument_type(list_arg["elts"][i]);
    if (current_type != element_type && !current_type.empty())
    {
      log_warning(
        "Mixed types detected in list literal: {} vs {}. Using 'list[int]' "
        "as fallback ({}:{})",
        element_type,
        current_type,
        python_filename_,
        current_line_);
      return "list[int]"; // Fallback for mixed numeric types
    }
  }

  // Return the full generic type notation
  return "list[" + element_type + "]";
}

template <class Json>
std::string python_annotation<Json>::get_type_from_binary_expr(
  const Json &stmt,
  const Json &body)
{
  std::string type("");

  const Json &lhs =
    stmt.contains("value") ? stmt["value"]["left"] : stmt["left"];

  if (lhs["_type"] == "BinOp")
    type = get_type_from_binary_expr(lhs, body);
  else if (lhs["_type"] == "List")
    type = "list";
  // Floor division (//) operations always result in an integer value
  else if (
    stmt.contains("value") && stmt["value"]["op"]["_type"] == "FloorDiv")
    type = "int";
  else
  {
    // If the LHS of the binary operation is a variable, its type is retrieved
    if (lhs["_type"] == "Name")
    {
      // Retrieve the type from the variable declaration within the current function body
      Json left_op = find_annotated_assign(lhs["id"], body["body"]);

      // If not found in the function body, try to retrieve it from the function arguments
      if (left_op.empty() && body.contains("args"))
        left_op = find_annotated_assign(lhs["id"], body["args"]["args"]);

      // Check current function scope for function parameters
      if (
        left_op.empty() && current_func != nullptr &&
        (*current_func).contains("args"))
      {
        left_op =
          find_annotated_assign(lhs["id"], (*current_func)["args"]["args"]);
      }

      if (
        !left_op.empty() && left_op.contains("annotation") &&
        left_op["annotation"].contains("id") &&
        left_op["annotation"]["id"].is_string())
      {
        type = left_op["annotation"]["id"];
      }
      // As a fallback, check global scope for prior annotation
      if (type.empty())
      {
        Json global_op = find_annotated_assign(lhs["id"], ast_["body"]);
        if (
          !global_op.empty() && global_op.contains("annotation") &&
          global_op["annotation"].contains("id") &&
          global_op["annotation"]["id"].is_string())
        {
          type = global_op["annotation"]["id"];
        }
      }
    }
    else if (lhs["_type"] == "UnaryOp")
    {
      const auto &operand = lhs["operand"];
      if (operand["_type"] == "Constant")
        type = get_type_from_constant(operand);
    }
    else if (lhs["_type"] == "Subscript")
    {
      // Handle subscript operations like dp[i-1], prices[i], etc.
      const std::string &var_name = lhs["value"]["id"];
      Json var_node =
        json_utils::find_var_decl(var_name, get_current_func_name(), ast_);

      if (!var_node.empty() && var_node.contains("annotation"))
      {
        std::string var_type;

        // Handle generic type annotations like list[int] (Subscript nodes)
        if (
          var_node["annotation"].contains("_type") &&
          var_node["annotation"]["_type"] == "Subscript" &&
          var_node["annotation"].contains("value") &&
          var_node["annotation"]["value"].contains("id"))
        {
          var_type = var_node["annotation"]["value"]["id"];

          // For list[T], return T directly from slice
          if (
            var_type == "list" && var_node["annotation"].contains("slice") &&
            var_node["annotation"]["slice"].contains("id"))
          {
            type = var_node["annotation"]["slice"]["id"];
          }
        }
        // Handle simple type annotations like int, str (Name nodes)
        else if (
          var_node["annotation"].contains("id") &&
          var_node["annotation"]["id"].is_string())
        {
          var_type = var_node["annotation"]["id"];

          // For list[T], return T. For other types, return the type itself
          if (var_type == "list")
          {
            // Try to get subtype from list initialization
            if (var_node.contains("value") && !var_node["value"].is_null())
            {
              std::string subtype = get_list_subtype(var_node["value"]);
              type = subtype.empty() ? "Any" : subtype;
            }
            else
              type = "Any"; // Unknown list element type
          }
          else
          {
            type = var_type;
          }
        }
      }
    }
    else if (lhs["_type"] == "Call" && lhs["func"]["_type"] == "Name")
    {
      // Handle function calls in binary expressions like float("1.1") + float("2.2")
      const std::string &func_name = lhs["func"]["id"];

      if (type_utils::is_builtin_type(func_name))
        type = func_name; // float() returns float, int() returns int, etc.
      else
        type = get_function_return_type(
          func_name, body); // For user-defined functions
    }
    else if (lhs["_type"] == "Constant")
      type = get_type_from_constant(lhs);
    else if (lhs["_type"] == "Call" && lhs["func"]["_type"] == "Attribute")
      type = get_type_from_method(lhs);
    else if (lhs["_type"] == "Attribute")
    {
      // Construct full attribute name (e.g., "string.digits")
      if (lhs["value"]["_type"] == "Name" && lhs["value"].contains("id"))
      {
        std::string full_name =
          lhs["value"]["id"].template get<std::string>() + "." +
          lhs["attr"].template get<std::string>();
        auto it = builtin_functions().find(full_name);
        if (it != builtin_functions().end())
          type = it->second;
      }
    }
  }

  // If still unknown, try RHS or fallback to Any for arithmetic ops
  if (type.empty())
  {
    const Json &rhs =
      stmt.contains("value") ? stmt["value"]["right"] : stmt["right"];

    if (rhs["_type"] == "Constant")
      type = get_type_from_constant(rhs);
    else if (rhs["_type"] == "Name")
    {
      Json right_op = find_annotated_assign(
        rhs["id"], body.contains("body") ? body["body"] : ast_["body"]);
      if (
        right_op.contains("annotation") &&
        right_op["annotation"].contains("id") &&
        right_op["annotation"]["id"].is_string())
        type = right_op["annotation"]["id"];
    }

    if (
      type.empty() && stmt.contains("value") &&
      stmt["value"].contains("op") && stmt["value"]["op"].contains("_type"))
    {
      type = "Any";
    }
  }

  return type;
}

template <class Json>
std::string
python_annotation<Json>::infer_lambda_return_type(const Json &lambda_elem) const
{
  const Json &lambda_body = lambda_elem["value"]["body"];
  if (lambda_body["_type"] == "BinOp")
    return "float"; // Match converter's default of double_type()
  else if (lambda_body["_type"] == "Compare")
    return "bool";
  else
    return "Any"; // Default for other lambda expressions
}

template <class Json>
std::string python_annotation<Json>::get_function_return_type(
  const std::string &func_name,
  const Json &ast)
{
  // Guard against infinite recursion
  if (functions_in_analysis_.count(func_name) > 0)
  {
    // Function is calling itself: try to infer from non-recursive return statements
    for (const Json &elem : ast["body"])
    {
      if (elem["_type"] == "FunctionDef" && elem["name"] == func_name)
      {
        // When override flag is set, always infer instead of using annotation
        bool should_override =
          config.options.get_bool_option("override-return-annotation");

        // Check if function has explicit return type annotation
        if (
          !should_override && elem.contains("returns") &&
          !elem["returns"].is_null() && elem["returns"].contains("id"))
        {
          functions_in_analysis_.erase(func_name);
          return elem["returns"]["id"];
        }

        // Try to infer from return statements (excluding recursive calls)
        std::string inferred =
          infer_from_return_statements(elem["body"], func_name);
        if (!inferred.empty())
        {
          functions_in_analysis_.erase(func_name);
          return inferred;
        }

        // No annotation and can't infer: return empty to avoid crash
        functions_in_analysis_.erase(func_name);
        return "";
      }
    }
    functions_in_analysis_.erase(func_name);
    return "";
  }

  // Add function to set before analysis
  functions_in_analysis_.insert(func_name);

  // Get type from nondet_<type> functions
  if (func_name.find("nondet_") == 0)
  {
    size_t last_underscore_pos = func_name.find_last_of('_');
    if (last_underscore_pos != std::string::npos)
    {
      functions_in_analysis_.erase(func_name);
      return func_name.substr(last_underscore_pos + 1);
    }
  }

  // Check override flag
  bool should_override =
    config.options.get_bool_option("override-return-annotation");

  // Search for function (including nested functions) using recursive helper
  Json func_elem = find_function_recursive(func_name, ast_["body"]);

  // Also check current function scope for local nested functions
  if (func_elem.empty() && current_func != nullptr)
  {
    func_elem = find_function_recursive(func_name, (*current_func)["body"]);
  }

  // Process the found function (if any)
  if (!func_elem.empty())
  {
    // If override is set, skip annotation and go straight to inference
    if (!should_override)
    {
      // Check if function has a return type annotation first (before inferring)
      if (func_elem.contains("returns") && !func_elem["returns"].is_null())
      {
        const auto &returns = func_elem["returns"];

        // Handle different return type annotation structures
        if (returns.contains("_type"))
        {
          const std::string &return_type = returns["_type"];

          // Handle Subscript type (e.g., List[int], Dict[str, int])
          if (return_type == "Subscript")
          {
            functions_in_analysis_.erase(func_name);
            if (returns.contains("value") && returns["value"].contains("id"))
              return returns["value"]["id"];
            else
              return "Any"; // Default for complex subscript types
          }
          // Handle Constant type (e.g., None)
          else if (return_type == "Constant")
          {
            functions_in_analysis_.erase(func_name);
            if (returns.contains("value") && returns["value"].is_null())
              return "NoneType";
            else if (
              returns.contains("value") && returns["value"].is_string())
            {
              // Forward reference annotation: -> "float", -> "MyClass", etc.
              // Validate that the string looks like a Python identifier (or
              // dotted name) before returning it, to guard against arbitrary
              // string constants used as non-type annotations.
              std::string type_name =
                returns["value"].template get<std::string>();
              bool valid = !type_name.empty() &&
                           (std::isalpha((unsigned char)type_name[0]) ||
                            type_name[0] == '_');
              for (size_t i = 1; valid && i < type_name.size(); ++i)
                valid = std::isalnum((unsigned char)type_name[i]) ||
                        type_name[i] == '_' || type_name[i] == '.';
              if (valid)
                return type_name;
              // Not a valid identifier — treat as opaque
              return "Any";
            }
            else if (returns.contains("value"))
              return "Any"; // Other constant types
          }
          // Handle other annotation types
          else
          {
            // Try to extract id if it exists
            if (returns.contains("id"))
            {
              // If the body also has `return None` paths, the function
              // actually returns Optional[T] regardless of the annotation.
              // Return "" so the converter handles this as Optional.
              if (has_return_none(func_elem["body"]))
              {
                functions_in_analysis_.erase(func_name);
                return "";
              }
              functions_in_analysis_.erase(func_name);
              return returns["id"];
            }
          }
        }
        // Handle case where returns exists but doesn't have expected structure
        else
        {
          log_warning(
            "Unrecognized return type annotation for function "
            "{}",
            func_name);
          functions_in_analysis_.erase(func_name);
          return "Any"; // Safe default
        }
      }
    }

    // Try to infer return type from actual return statements
    // Use recursive inference to find return types in all blocks
    std::string inferred_type =
      infer_from_return_statements(func_elem["body"], func_name);

    if (!inferred_type.empty() && inferred_type != "NoneType")
    {
      // Found a non-None return type; check for mixed value+None returns
      if (has_return_none(func_elem["body"]))
      {
        // Mixed returns: leave unannotated so converter handles via Optional
        functions_in_analysis_.erase(func_name);
        return "";
      }
      functions_in_analysis_.erase(func_name);
      return inferred_type;
    }

    // Check top-level returns as fallback
    auto return_node = json_utils::find_return_node(func_elem["body"]);
    if (!return_node.empty())
    {
      std::string fallback_type;
      try
      {
        infer_type(return_node, func_elem, fallback_type);
      }
      catch (std::runtime_error &)
      {
        // Return value type could not be inferred (e.g. call through a
        // function-pointer parameter); leave fallback_type empty.
      }
      functions_in_analysis_.erase(func_name);
      return fallback_type;
    }

    // If function has no explicit return statements, assume void/None
    functions_in_analysis_.erase(func_name);
    return "NoneType";
  }

  // Check for lambda assignments when regular function not found
  Json lambda_elem = find_lambda_in_body(func_name, ast["body"]);

  // Also check current function scope if not found globally
  if (lambda_elem.empty() && current_func != nullptr)
    lambda_elem = find_lambda_in_body(func_name, (*current_func)["body"]);

  if (!lambda_elem.empty())
  {
    functions_in_analysis_.erase(func_name);
    return infer_lambda_return_type(lambda_elem);
  }

  // Get type from imported functions
  try
  {
    if (module_manager_)
    {
      const auto &import_node =
        json_utils::find_imported_function(ast_, func_name);
      auto module = module_manager_->get_module(import_node["module"]);

      if (!module)
        throw std::runtime_error("module not found");

      // Try to get it as a function first
      try
      {
        auto func_info = module->get_function(func_name);

        // If return_type is empty or "NoneType", check if it's actually a class
        if (
          func_info.return_type_.empty() ||
          func_info.return_type_ == "NoneType")
        {
          // It might be a class constructor (__init__ returns None)
          // Return the class name as the type
          functions_in_analysis_.erase(func_name);
          return func_name;
        }

        functions_in_analysis_.erase(func_name);
        return func_info.return_type_;
      }
      catch (std::runtime_error &)
      {
        // If get_function fails, it might be a class
        functions_in_analysis_.erase(func_name);
        return func_name;
      }
    }
  }
  catch (std::runtime_error &)
  {
  }

  // Probe wildcard imports (`from X import *`) for @p func_name when the
  // named-import lookup above missed. find_imported_function only matches
  // explicit aliases, so star-imported functions stay invisible to the
  // annotator's type inference without this fallback (GitHub #4564).
  if (std::string t = resolve_wildcard_import_func(func_name); !t.empty())
  {
    functions_in_analysis_.erase(func_name);
    return t;
  }

  // Check if the function is a built-in function
  auto it = builtin_functions().find(func_name);
  if (it != builtin_functions().end())
  {
    functions_in_analysis_.erase(func_name);
    return it->second;
  }

  // Check if the name is a class constructor
  // (e.g., A() returns an A instance)
  {
    const std::string resolved =
      json_utils::get_object_alias(ast_, func_name);
    if (json_utils::is_class(resolved, ast_))
    {
      functions_in_analysis_.erase(func_name);
      return resolved;
    }
  }

  functions_in_analysis_.erase(func_name);

  std::ostringstream oss;
  oss << "Function \"" << func_name << "\" not found (" << python_filename_
      << " line " << current_line_ << ")";

  throw std::runtime_error(oss.str());
}
