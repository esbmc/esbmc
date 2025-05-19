#pragma once

#include <python-frontend/json_utils.h>
#include <python-frontend/global_scope.h>
#include <python-frontend/module_manager.h>
#include <python-frontend/module.h>
#include <python-frontend/type_utils.h>
#include <util/message.h>

#include <string>

enum class InferResult
{
  OK,
  UNKNOWN,
};

template <class Json>
class python_annotation
{
public:
  python_annotation(Json &ast, global_scope &gs)
    : ast_(ast), gs_(gs), current_func(nullptr), current_line_(0)
  {
    python_filename_ = ast_["filename"].template get<std::string>();
    if (ast_.contains("ast_output_dir"))
      module_manager_ =
        module_manager::create(ast_["ast_output_dir"], python_filename_);
  }

  void add_type_annotation()
  {
    // Add type annotations to global scope variables
    annotate_global_scope();
    current_line_ = 0;

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
    current_line_ = 0;

    for (Json &elem : ast_["body"])
    {
      if (elem["_type"] == "FunctionDef" && elem["name"] == func_name)
      {
        get_global_elements(elem["body"]);
        // Add type annotations to global scope variables
        if (!referenced_global_elements.empty())
          annotate_global_scope();
        filter_global_elements_ = false;

        // Add annotation to a specific function
        annotate_function(elem);
        return;
      }
    }
  }

private:
  /* Get the global elements referenced by a function */
  void get_global_elements(const Json &node)
  {
    // Checks if the current node is a variable identifier
    if (
      node.contains("_type") && node["_type"] == "Name" && node.contains("id"))
    {
      const std::string &var_name = node["id"];
      Json var_node = json_utils::find_var_decl(var_name, "", ast_);
      if (!var_node.empty())
      {
        gs_.add_variable(var_name);
        referenced_global_elements.push_back(var_node);
      }
    }

    if (
      node.contains("_type") && node["_type"] == "Call" &&
      node.contains("func") && node["func"]["_type"] == "Name")
    {
      const std::string &class_name = node["func"]["id"];
      // Checks if the current node is a constructor call
      Json class_node = json_utils::find_class(ast_["body"], class_name);
      if (!class_node.empty())
      {
        gs_.add_class(class_name);
        referenced_global_elements.push_back(class_node);
      }
      else
      {
        const auto &func_name = node["func"]["id"];
        if (!type_utils::is_builtin_type(func_name))
        {
          try
          {
            const auto func_node =
              json_utils::find_function(ast_["body"], func_name);
            get_global_elements(func_node);
          }
          catch (std::runtime_error &)
          {
          }
        }
      }
    }

    // Recursively iterates through all fields of the node if it is an object
    if (node.is_object())
    {
      for (auto it = node.begin(); it != node.end(); ++it)
        get_global_elements(it.value());
    }

    // Iterates over the elements if the current node is an array
    else if (node.is_array())
    {
      for (const auto &element : node)
        get_global_elements(element);
    }
    filter_global_elements_ = true;
  }

  void annotate_global_scope()
  {
    add_annotation(ast_);
  }

  void annotate_function(Json &function_element)
  {
    current_func = &function_element;

    // Add type annotations within the function
    add_annotation(function_element);

    auto return_node = json_utils::find_return_node(function_element["body"]);
    if (
      !return_node.empty() &&
      (function_element["returns"].is_null() ||
       config.options.get_bool_option("override-return-annotation")))
    {
      std::string inferred_type;
      if (
        infer_type(return_node, function_element, inferred_type) ==
        InferResult::OK)
      {
        // Update the function node to include the return type annotation
        function_element["returns"] = {
          {"_type", "Name"},
          {"id", inferred_type},
          {"ctx", {{"_type", "Load"}}},
          {"lineno", function_element["lineno"]},
          {"col_offset", function_element["col_offset"]},
          {"end_lineno", function_element["lineno"]},
          {"end_col_offset",
           function_element["col_offset"].template get<int>() +
             inferred_type.size()}};
      }
    }

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

  std::string get_current_func_name()
  {
    if (!current_func)
      return std::string();

    return (*current_func)["name"];
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
    else if (lhs["_type"] == "List")
    {
      type = "list";
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
      else if (lhs["_type"] == "Call" && lhs["func"]["_type"] == "Attribute")
        type = get_type_from_method(lhs);
    }

    return type;
  }

  std::string
  get_function_return_type(const std::string &func_name, const Json &ast)
  {
    // Get type from nondet_<type> functions
    if (func_name.find("nondet_") == 0)
    {
      size_t last_underscore_pos = func_name.find_last_of('_');
      if (last_underscore_pos != std::string::npos)
      {
        // Return the substring from the position after the last underscore to the end
        return func_name.substr(last_underscore_pos + 1);
      }
    }

    for (const Json &elem : ast["body"])
    {
      if (elem["_type"] == "FunctionDef" && elem["name"] == func_name)
      {
        auto return_node = json_utils::find_return_node(elem["body"]);
        if (!return_node.empty())
        {
          std::string inferred_type;
          infer_type(return_node, elem, inferred_type);
          return inferred_type;
        }

        if (elem["returns"]["_type"] == "Subscript")
          return elem["returns"]["value"]["id"];
        else
          return elem["returns"]["id"];
      }
    }

    // Get type from imported functions
    try
    {
      if (module_manager_)
      {
        const auto &import_node =
          json_utils::find_imported_function(ast_, func_name);
        auto module = module_manager_->get_module(import_node["module"]);
        return module->get_function(func_name).return_type_;
      }
    }
    catch (std::runtime_error &)
    {
    }

    std::ostringstream oss;
    oss << "Function \"" << func_name << "\" not found (" << python_filename_
        << " line " << current_line_ << ")";

    throw std::runtime_error(oss.str());
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

  std::string get_type_from_json(const Json &value)
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

  std::string get_list_subtype(const Json &list)
  {
    std::string list_subtype;

    if (list["_type"] == "Call" && list["func"]["attr"] == "array")
      return get_list_subtype(list["args"][0]);

    if (!list.contains("elts"))
      return "";

    if (!list["elts"].empty())
      list_subtype = get_type_from_json(list["elts"][0]["value"]);

    for (const auto &elem : list["elts"])
    {
      if (get_type_from_json(elem["value"]) != list_subtype)
      {
        throw std::runtime_error("Multiple typed lists are not supported\n");
      }
    }
    return list_subtype;
  }

  std::string get_type_from_rhs_variable(const Json &element, const Json &body)
  {
    const auto &value_type = element["value"]["_type"];
    std::string rhs_var_name;

    if (value_type == "Name")
      rhs_var_name = element["value"]["id"];
    else if (value_type == "UnaryOp")
      rhs_var_name = element["value"]["operand"]["id"];
    else
      rhs_var_name = element["value"]["value"]["id"];

    assert(!rhs_var_name.empty());

    // Find RHS variable declaration in the current scope (e.g.: while/if block)
    Json rhs_node = find_annotated_assign(rhs_var_name, body["body"]);

    // Try to infer variable from current scope
    if (rhs_node.empty())
    {
      auto var = json_utils::get_var_node(rhs_var_name, body);
      std::string type;
      if (infer_type(var, body, type) == InferResult::OK && !type.empty())
      {
        return type;
      }
    }

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
      const auto &lineno = element["lineno"].template get<int>();

      std::ostringstream oss;
      oss << "Type inference failed at line " << lineno;
      oss << ". Variable " << rhs_var_name << " not found";

      throw std::runtime_error(oss.str());
    }

    if (value_type == "Subscript")
    {
      return get_list_subtype(rhs_node["value"]);
    }

    return rhs_node["annotation"]["id"];
  }

  std::string get_type_from_call(const Json &element)
  {
    const std::string &func_id = element["value"]["func"]["id"];

    if (
      json_utils::is_class<Json>(func_id, ast_) ||
      type_utils::is_builtin_type(func_id) ||
      type_utils::is_consensus_type(func_id))
      return func_id;

    if (type_utils::is_consensus_func(func_id))
      return type_utils::get_type_from_consensus_func(func_id);

    if (!type_utils::is_python_model_func(func_id))
      return get_function_return_type(func_id, ast_);

    return "";
  }

  std::string invert_substrings(const std::string &input)
  {
    std::vector<std::string> substrings;
    std::string token;
    std::istringstream stream(input);

    // Split the string using "." as the delimiter
    while (std::getline(stream, token, '.'))
    {
      substrings.push_back(token);
    }

    // Reverse the order of the substrings
    std::reverse(substrings.begin(), substrings.end());

    // Rebuild the string with "." between the reversed substrings
    std::string result;
    for (size_t i = 0; i < substrings.size(); ++i)
    {
      if (i != 0)
      {
        result += "."; // Add the dot separator
      }
      result += substrings[i];
    }

    return result;
  }

  std::string get_object_name(const Json &call, const std::string &prefix)
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
    obj_name += call["value"]["id"].template get<std::string>();
    if (obj_name.find('.') != std::string::npos)
      obj_name = invert_substrings(obj_name);

    return json_utils::get_object_alias(ast_, obj_name);
  }

  std::string get_type_from_method(const Json &call)
  {
    std::string type("");

    const std::string &obj = get_object_name(call["func"], std::string());

    // Get type from imported module
    if (module_manager_)
    {
      auto module = module_manager_->get_module(obj);
      if (module)
        return module->get_function(call["func"]["attr"]).return_type_;
    }

    Json obj_node =
      json_utils::find_var_decl(obj, get_current_func_name(), ast_);

    if (obj_node.empty())
      throw std::runtime_error("Object \"" + obj + "\" not found.");

    const std::string &obj_type =
      obj_node["annotation"]["id"].template get<std::string>();

    if (type_utils::is_builtin_type(obj_type))
      type = obj_type;

    return type;
  }

  InferResult
  infer_type(const Json &stmt, const Json &body, std::string &inferred_type)
  {
    if (stmt.empty())
      return InferResult::UNKNOWN;

    if (stmt["_type"] == "arg")
    {
      if (stmt["annotation"].contains("value"))
        inferred_type =
          stmt["annotation"]["value"]["id"].template get<std::string>();
      else
        inferred_type = stmt["annotation"]["id"].template get<std::string>();
      return InferResult::OK;
    }

    const auto &value_type = stmt["value"]["_type"];

    // Get type from RHS constant
    if (value_type == "Constant")
    {
      inferred_type = get_type_from_constant(stmt["value"]);
    }
    else if (value_type == "List")
    {
      inferred_type = "list";
    }
    else if (value_type == "UnaryOp") // Handle negative numbers
    {
      const auto &operand = stmt["value"]["operand"];
      const auto &operand_type = operand["_type"];
      if (operand_type == "Constant")
        inferred_type = get_type_from_constant(operand);
      else if (operand_type == "Name")
        inferred_type = get_type_from_rhs_variable(stmt, body);
    }

    // Get type from RHS variable
    else if (value_type == "Name" || value_type == "Subscript")
    {
      inferred_type = get_type_from_rhs_variable(stmt, body);
    }

    // Get type from RHS binary expression
    else if (value_type == "BinOp")
    {
      std::string got_type = get_type_from_binary_expr(stmt, body);
      if (!got_type.empty())
        inferred_type = got_type;
    }

    // Get type from top-level functions
    else if (
      value_type == "Call" && stmt["value"]["func"]["_type"] == "Name" &&
      !type_utils::is_python_model_func(stmt["value"]["func"]["id"]))
    {
      inferred_type = get_type_from_call(stmt);
    }

    // Get type from methods
    else if (
      value_type == "Call" && stmt["value"]["func"]["_type"] == "Attribute")
    {
      inferred_type = get_type_from_method(stmt["value"]);
    }
    else
      return InferResult::UNKNOWN;

    return InferResult::OK;
  }

  void add_annotation(Json &body)
  {
    for (auto &element : body["body"])
    {
      auto itr = std::find(
        referenced_global_elements.begin(),
        referenced_global_elements.end(),
        element);

      if (filter_global_elements_ && itr == referenced_global_elements.end())
        continue;

      if (element.contains("lineno"))
        current_line_ = element["lineno"].template get<int>();

      auto &stmt_type = element["_type"];

      if (stmt_type == "If" || stmt_type == "While")
      {
        add_annotation(element);
        continue;
      }

      const std::string function_flag = config.options.get_option("function");
      if (!function_flag.empty())
      {
        if (
          stmt_type == "Expr" && element.contains("value") &&
          element["value"]["_type"] == "Call" &&
          element["value"]["func"]["_type"] == "Name")
        {
          auto &func_node = json_utils::find_function(
            ast_["body"], element["value"]["func"]["id"]);
          if (!func_node.empty())
            add_annotation(func_node);
        }
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

      if (infer_type(element, body, inferred_type) == InferResult::UNKNOWN)
        continue;

      if (inferred_type.empty())
      {
        std::ostringstream oss;
        oss << "Type inference failed for "
            << stmt_type.template get<std::string>() << " at line "
            << current_line_;

        throw std::runtime_error(oss.str());
      }

      update_assignment_node(element, inferred_type);
      if (itr != referenced_global_elements.end())
        *itr = element;
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
    else if (target.contains("slice"))
    {
      return; // No need to annotate assignments to array elements.
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
    auto update_offsets = [&inferred_type](Json &value) {
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
  global_scope &gs_;
  std::shared_ptr<module_manager> module_manager_;
  Json *current_func;
  int current_line_;
  std::string python_filename_;
  bool filter_global_elements_ = false;
  std::vector<Json> referenced_global_elements;
};
