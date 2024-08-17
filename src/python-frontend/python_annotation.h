#pragma once

#include <python-frontend/json_utils.h>
#include <python-frontend/python_frontend_types.h>
#include <util/message.h>

#include <string>

template <class Json>
class python_annotation
{
public:
  python_annotation(Json &ast) : ast_(ast), current_func(nullptr)
  {
  }

  void add_type_annotation()
  {
    // Add type annotations to global scope variables
    annotate_global_scope();

    // Add type annotation to all functions and class methods
    for (Json &element : ast_["body"])
    {
      // Process top-level functions
      if (element["_type"] == "FunctionDef")
      {
        annotate_function(element);
      }
      // Process classes and their methods
      else if (element["_type"] == "ClassDef")
      {
        annotate_class(element);
      }
    }
  }

  void add_type_annotation(const std::string &func_name)
  {
    // Add type annotations to global scope variables
    annotate_global_scope();

    for (Json &elem : ast_["body"])
    {
      if (elem["_type"] == "FunctionDef" && elem["name"] == func_name)
      {
        // Add annotation to a specific function
        annotate_function(elem);
        return;
      }
    }
  }

private:
  void annotate_global_scope()
  {
    add_annotation(ast_);
  }

  void annotate_function(Json &function_element)
  {
    current_func = &function_element;

    // Add type annotations within the function
    add_annotation(function_element);

    // Update the end column offset after adding annotations
    update_end_col_offset(function_element);

    current_func = nullptr;
  }

  void annotate_class(Json &class_element)
  {
    for (Json &class_member : class_element["body"])
    {
      // Process methods in the class
      if (class_member["_type"] == "FunctionDef")
      {
        // Add type annotations within the class member function
        annotate_function(class_member);
      }
    }
  }

  std::string get_type_from_constant(const Json &element)
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

    return std::string();
  }

  std::string get_type_from_binary_expr(const Json &stmt, const Json &body)
  {
    std::string type("");

    const Json &lhs =
      stmt.contains("value") ? stmt["value"]["left"] : stmt["left"];

    if (lhs["_type"] == "BinOp")
    {
      type = get_type_from_binary_expr(lhs, body);
    }
    // Floor division (//) operations always result in an integer value
    else if (
      stmt.contains("value") && stmt["value"]["op"]["_type"] == "FloorDiv")
    {
      type = "int";
    }
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

        if (!left_op.empty())
          type = left_op["annotation"]["id"];
      }
      else if (lhs["_type"] == "Constant")
        type = get_type_from_constant(lhs);
    }

    return type;
  }

  std::string
  get_function_return_type(const std::string &func_name, const Json &ast)
  {
    for (const Json &elem : ast["body"])
    {
      if (
        elem["_type"] == "FunctionDef" && elem["name"] == func_name &&
        elem.contains("returns") && !elem["returns"].is_null())
      {
        return elem["returns"]["id"];
      }
    }
    return std::string();
  }

  std::string get_type_from_lhs(const std::string &id, Json &body)
  {
    // Search for LHS annotation in the current scope (e.g. while/if block)
    Json node = find_annotated_assign(id, body["body"]);

    // Fall back to the current function
    if (node.empty() && current_func != nullptr)
      node = find_annotated_assign(id, (*current_func)["body"]);

    // Fall back to variables in the global scope
    if (node.empty())
      node = find_annotated_assign(id, ast_["body"]);

    return node.empty() ? "" : node["annotation"]["id"];
  }

  std::string get_type_from_rhs_variable(const Json &element, Json &body)
  {
    const auto &value_type = element["value"]["_type"];
    std::string rhs_var_name = value_type == "Name"
                                 ? element["value"]["id"]
                                 : element["value"]["value"]["id"];

    // Find RHS variable declaration in the current scope (e.g.: while/if block)
    Json rhs_node = find_annotated_assign(rhs_var_name, body["body"]);

    // Find RHS variable declaration in the current function
    if (rhs_node.empty() && current_func)
      rhs_node = find_annotated_assign(rhs_var_name, (*current_func)["body"]);

    // Find RHS variable in the current function args
    if (rhs_node.empty() && body.contains("args"))
      rhs_node = find_annotated_assign(rhs_var_name, body["args"]["args"]);

    // Find RHS variable node in the global scope
    if (rhs_node.empty())
      rhs_node = find_annotated_assign(rhs_var_name, ast_["body"]);

    if (rhs_node.empty())
    {
      log_error("Variable {} not found.", rhs_var_name.c_str());
      abort();
    }

    return rhs_node["annotation"]["id"];
  }

  std::string get_type_from_call(const Json &element)
  {
    const std::string &func_id = element["value"]["func"]["id"];

    if (
      json_utils::is_class<Json>(func_id, ast_) || is_builtin_type(func_id) ||
      is_consensus_type(func_id))
      return func_id;

    if (is_consensus_func(func_id))
      return get_type_from_consensus_func(func_id);

    if (!is_model_func(func_id))
      return get_function_return_type(func_id, ast_);

    return "";
  }

  void add_annotation(Json &body)
  {
    for (auto &element : body["body"])
    {
      auto &stmt_type = element["_type"];

      if (stmt_type == "If" || stmt_type == "While")
      {
        add_annotation(element);
        continue;
      }

      if (stmt_type != "Assign" || !element["type_comment"].is_null())
        continue;

      std::string inferred_type("");

      // Check if LHS was previously annotated
      if (
        element.contains("targets") && element["targets"][0]["_type"] == "Name")
      {
        inferred_type = get_type_from_lhs(element["targets"][0]["id"], body);
      }

      const auto &value_type = element["value"]["_type"];

      // Get type from RHS constant
      if (value_type == "Constant")
      {
        inferred_type = get_type_from_constant(element["value"]);
      }
      else if (value_type == "List")
      {
        inferred_type = "list";
      }
      else if (
        value_type == "UnaryOp" && element["value"]["operand"]["_type"] ==
                                     "Constant") // Handle negative numbers
      {
        inferred_type = get_type_from_constant(element["value"]["operand"]);
      }

      // Get type from RHS variable
      else if (value_type == "Name" || value_type == "Subscript")
      {
        inferred_type = get_type_from_rhs_variable(element, body);
      }

      // Get type from RHS binary expression
      else if (value_type == "BinOp")
      {
        std::string got_type = get_type_from_binary_expr(element, body);
        if (!got_type.empty())
          inferred_type = got_type;
      }

      else if (
        value_type == "Call" && !is_model_func(element["value"]["func"]["id"]))
      {
        inferred_type = get_type_from_call(element);
      }
      else
        continue;

      if (inferred_type.empty())
      {
        log_error("Type undefined for:\n{}", element.dump(2).c_str());
        abort();
      }

      update_assignment_node(element, inferred_type);
    }
  }

  void update_assignment_node(Json &element, const std::string &inferred_type)
  {
    // Update type field
    element["_type"] = "AnnAssign";

    auto target = element["targets"][0];
    std::string id;

    // Determine the ID based on the target type
    if (target.contains("id"))
    {
      id = target["id"];
    }
    // Get LHS from members access on assignments. e.g.: x.data = 10
    else if (target["_type"] == "Attribute")
    {
      id = target["value"]["id"].template get<std::string>() + "." +
           target["attr"].template get<std::string>();
    }

    assert(!id.empty());

    // Calculate column offset
    int col_offset = target["col_offset"].template get<int>() + id.size() + 1;

    // Create the annotation field
    element["annotation"] = {
      {"_type", target["_type"]},
      {"col_offset", col_offset},
      {"ctx", {{"_type", "Load"}}},
      {"end_col_offset", col_offset + inferred_type.size()},
      {"end_lineno", target["end_lineno"]},
      {"id", inferred_type},
      {"lineno", target["lineno"]}};

    // Update element properties
    element["end_col_offset"] =
      element["end_col_offset"].template get<int>() + inferred_type.size() + 1;
    element["end_lineno"] = element["lineno"];
    element["simple"] = 1;

    // Replace "targets" array with "target" object
    element["target"] = std::move(target);
    element.erase("targets");

    // Remove unnecessary field
    element.erase("type_comment");

    // Update value fields with the correct offsets
    auto update_offsets = [&](Json &value) {
      value["col_offset"] =
        value["col_offset"].template get<int>() + inferred_type.size() + 1;
      value["end_col_offset"] =
        value["end_col_offset"].template get<int>() + inferred_type.size() + 1;
    };

    update_offsets(element["value"]);

    // Adjust column offsets for function calls with arguments
    if (element["value"].contains("args"))
    {
      for (auto &arg : element["value"]["args"])
      {
        update_offsets(arg);
      }
    }

    // Adjust column offsets in function call node
    if (element["value"].contains("func"))
    {
      update_offsets(element["value"]["func"]);
    }
  }

  void update_end_col_offset(Json &ast)
  {
    int max_col_offset = ast["end_col_offset"];
    for (auto &elem : ast["body"])
    {
      if (elem["end_col_offset"] > max_col_offset)
        max_col_offset = elem["end_col_offset"];
    }
    ast["end_col_offset"] = max_col_offset;
  }

  const Json
  find_annotated_assign(const std::string &node_name, const Json &body) const
  {
    for (const Json &elem : body)
    {
      if (
        elem.contains("_type") &&
        ((elem["_type"] == "AnnAssign" && elem.contains("target") &&
          elem["target"].contains("id") &&
          elem["target"]["id"].template get<std::string>() == node_name) ||
         (elem["_type"] == "arg" && elem["arg"] == node_name)))
      {
        return elem;
      }
    }
    return Json();
  }

  Json &ast_;
  Json *current_func;
};
