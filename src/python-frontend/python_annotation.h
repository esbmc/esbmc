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

// Handle Python built-in functions
static const std::map<std::string, std::string> builtin_functions = {
  // Type conversion functions
  {"int", "int"},
  {"float", "float"},
  {"str", "str"},
  {"bool", "bool"},
  {"list", "list"},
  {"dict", "dict"},
  {"set", "set"},
  {"tuple", "tuple"},

  // Numeric functions
  {"abs", "int"},   // Can return int or float, but int is common case
  {"round", "int"}, // Can return int or float
  {"min", "Any"},   // Type depends on input
  {"max", "Any"},   // Type depends on input
  {"sum", "int"},   // Can return int or float, but int is common case
  {"pow", "int"},   // Can return int or float

  // Sequence functions
  {"len", "int"},
  {"range", "range"},
  {"enumerate", "enumerate"},
  {"zip", "zip"},
  {"reversed", "reversed"},
  {"sorted", "list"},

  // I/O functions
  {"print", "NoneType"},
  {"input", "str"},
  {"open", "file"},

  // Utility functions
  {"isinstance", "bool"},
  {"issubclass", "bool"},
  {"hasattr", "bool"},
  {"getattr", "Any"},
  {"setattr", "NoneType"},
  {"delattr", "NoneType"},
  {"callable", "bool"},
  {"id", "int"},
  {"hash", "int"},
  {"repr", "str"},
  {"ascii", "str"},
  {"ord", "int"},
  {"chr", "str"},
  {"bin", "str"},
  {"oct", "str"},
  {"hex", "str"},
  {"format", "str"},

  // Iteration functions
  {"iter", "iterator"},
  {"next", "Any"},
  {"all", "bool"},
  {"any", "bool"},
  {"filter", "filter"},
  {"map", "map"},

  // Variable functions
  {"vars", "dict"},
  {"dir", "list"},
  {"globals", "dict"},
  {"locals", "dict"},

  // Execution functions
  {"eval", "Any"},
  {"exec", "NoneType"},
  {"compile", "code"},

  // Import functions
  {"__import__", "module"}};

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
        annotate_function(element);
      // Process classes and their methods
      else if (element["_type"] == "ClassDef")
        annotate_class(element);
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
  // Method to infer and annotate unannotated function parameters
  void infer_parameter_types(Json &function_element)
  {
    const std::string &func_name = function_element["name"];

    // Find all calls to this function in the AST
    std::vector<Json> function_calls =
      find_function_calls(func_name, ast_["body"]);

    if (function_calls.empty())
      return; // No calls found, cannot infer

    // For each parameter, try to infer its type from the function calls
    if (
      function_element.contains("args") &&
      function_element["args"].contains("args"))
    {
      Json &params = function_element["args"]["args"];

      for (size_t i = 0; i < params.size(); ++i)
      {
        Json &param = params[i];

        // Skip if parameter is already annotated
        if (param.contains("annotation") && !param["annotation"].is_null())
          continue;

        // Try to infer type from function calls
        std::string inferred_type =
          infer_parameter_type_from_calls(i, function_calls);

        if (!inferred_type.empty())
        {
          // Add annotation to parameter
          add_parameter_annotation(param, inferred_type);
        }
      }
    }
  }

  // Method to find all function calls to a specific function
  std::vector<Json>
  find_function_calls(const std::string &func_name, const Json &body)
  {
    std::vector<Json> calls;
    find_function_calls_recursive(func_name, body, calls);
    return calls;
  }

  // Recursive helper to find function calls
  void find_function_calls_recursive(
    const std::string &func_name,
    const Json &node,
    std::vector<Json> &calls)
  {
    if (node.is_object())
    {
      // Check if this is a function call to our target function
      if (
        node.contains("_type") && node["_type"] == "Call" &&
        node.contains("func") && node["func"]["_type"] == "Name" &&
        node["func"]["id"] == func_name)
      {
        calls.push_back(node);
      }

      // Recursively search all fields
      for (auto it = node.begin(); it != node.end(); ++it)
        find_function_calls_recursive(func_name, it.value(), calls);
    }
    else if (node.is_array())
    {
      for (const auto &element : node)
        find_function_calls_recursive(func_name, element, calls);
    }
  }

  // Method to infer parameter type from function calls
  std::string infer_parameter_type_from_calls(
    size_t param_index,
    const std::vector<Json> &function_calls)
  {
    std::string inferred_type;

    for (const Json &call : function_calls)
    {
      if (call.contains("args") && param_index < call["args"].size())
      {
        const Json &arg = call["args"][param_index];
        std::string arg_type = get_argument_type(arg);

        if (!arg_type.empty())
        {
          if (inferred_type.empty())
            inferred_type = arg_type;
          else if (inferred_type != arg_type)
          {
            // Type conflict between calls, use int as safe fallback
            log_warning(
              "Type inference conflict for parameter {}: {} vs {}. Using 'int' "
              "as fallback ({}:{})",
              param_index,
              inferred_type,
              arg_type,
              python_filename_,
              current_line_);
            return "int";
          }
        }
      }
    }

    return inferred_type;
  }

  // Method to get the type of an argument in a function call
  std::string get_argument_type(const Json &arg)
  {
    if (arg["_type"] == "Constant")
      return get_type_from_constant(arg);
    else if (arg["_type"] == "UnaryOp")
    {
      // Handle unary operations like -5, +3, not True
      if (arg.contains("operand"))
        return get_argument_type(arg["operand"]);
    }
    else if (arg["_type"] == "Name")
    {
      // Try to find the type of the variable
      Json var_node =
        json_utils::find_var_decl(arg["id"], get_current_func_name(), ast_);
      if (
        !var_node.empty() && var_node.contains("annotation") &&
        !var_node["annotation"].is_null() &&
        var_node["annotation"].contains("id"))
      {
        return var_node["annotation"]["id"];
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

    return ""; // Cannot determine type
  }

  // Method to get the full type of a list literal
  std::string get_list_type_from_literal(const Json &list_arg)
  {
    if (!list_arg.contains("elts") || list_arg["elts"].empty())
    {
      log_warning(
        "Empty or malformed list literal detected. Using 'list[int]' as "
        "default ({}:{})",
        python_filename_,
        current_line_);
      return "list[int]"; // Default fallback
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

  // Method to add annotation to a parameter
  void add_parameter_annotation(Json &param, const std::string &type)
  {
    int col_offset = param["col_offset"].template get<int>() +
                     param["arg"].template get<std::string>().length() + 1;

    // Check if this is a generic type (e.g., list[int])
    size_t bracket_pos = type.find('[');
    if (bracket_pos != std::string::npos)
    {
      // Extract base type and element type
      std::string base_type = type.substr(0, bracket_pos);
      std::string element_type =
        type.substr(bracket_pos + 1, type.length() - bracket_pos - 2);

      // Create Subscript annotation for generic types
      param["annotation"] = {
        {"_type", "Subscript"},
        {"value",
         {{"_type", "Name"},
          {"id", base_type},
          {"ctx", {{"_type", "Load"}}},
          {"lineno", param["lineno"]},
          {"col_offset", col_offset},
          {"end_lineno", param["lineno"]},
          {"end_col_offset", col_offset + base_type.size()}}},
        {"slice",
         {{"_type", "Name"},
          {"id", element_type},
          {"ctx", {{"_type", "Load"}}},
          {"lineno", param["lineno"]},
          {"col_offset", col_offset + base_type.size() + 1},
          {"end_lineno", param["lineno"]},
          {"end_col_offset",
           col_offset + base_type.size() + 1 + element_type.size()}}},
        {"ctx", {{"_type", "Load"}}},
        {"lineno", param["lineno"]},
        {"col_offset", col_offset},
        {"end_lineno", param["lineno"]},
        {"end_col_offset", col_offset + type.size()}};
    }
    else
    {
      // Create simple Name annotation for basic types
      param["annotation"] = {
        {"_type", "Name"},
        {"id", type},
        {"ctx", {{"_type", "Load"}}},
        {"lineno", param["lineno"]},
        {"col_offset", col_offset},
        {"end_lineno", param["lineno"]},
        {"end_col_offset", col_offset + type.size()}};
    }

    // Update the parameter's end_col_offset to account for the annotation
    param["end_col_offset"] =
      param["end_col_offset"].template get<int>() + type.size() + 1;
  }

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
            const auto &func_node =
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

    // Infer types for unannotated parameters based on function calls
    infer_parameter_types(function_element);

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
    if (rhs.is_null())
      return "NoneType";

    return std::string();
  }

  std::string get_type_from_binary_expr(const Json &stmt, const Json &body)
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
      }
      else if (lhs["_type"] == "Subscript")
      {
        // Handle subscript operations like dp[i-1], prices[i], etc.
        const std::string &var_name = lhs["value"]["id"];
        Json var_node =
          json_utils::find_var_decl(var_name, get_current_func_name(), ast_);

        if (!var_node.empty())
        {
          const std::string &var_type = var_node["annotation"]["id"];

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
    }

    return type;
  }

  Json find_lambda_in_body(const std::string &func_name, const Json &body) const
  {
    for (const Json &elem : body)
    {
      if (
        elem["_type"] == "Assign" && !elem["targets"].empty() &&
        elem["targets"][0].contains("id") &&
        elem["targets"][0]["id"] == func_name && elem.contains("value") &&
        elem["value"]["_type"] == "Lambda")
      {
        return elem;
      }
    }
    return Json();
  }

  std::string infer_lambda_return_type(const Json &lambda_elem) const
  {
    const Json &lambda_body = lambda_elem["value"]["body"];
    if (lambda_body["_type"] == "BinOp")
      return "float"; // Match converter's default of double_type()
    else if (lambda_body["_type"] == "Compare")
      return "bool";
    else
      return "Any"; // Default for other lambda expressions
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

    // Search the top-level AST body for a matching function definition
    for (const Json &elem : ast["body"])
    {
      if (elem["_type"] == "FunctionDef" && elem["name"] == func_name)
      {
        // Try to infer return type from actual return statements
        auto return_node = json_utils::find_return_node(elem["body"]);
        if (!return_node.empty())
        {
          std::string inferred_type;
          infer_type(return_node, elem, inferred_type);
          return inferred_type;
        }

        // Check if function has a return type annotation
        if (elem.contains("returns") && !elem["returns"].is_null())
        {
          const auto &returns = elem["returns"];

          // Handle different return type annotation structures
          if (returns.contains("_type"))
          {
            const std::string &return_type = returns["_type"];

            // Handle Subscript type (e.g., List[int], Dict[str, int])
            if (return_type == "Subscript")
            {
              if (returns.contains("value") && returns["value"].contains("id"))
                return returns["value"]["id"];
              else
                return "Any"; // Default for complex subscript types
            }
            // Handle Constant type (e.g., None)
            else if (return_type == "Constant")
            {
              if (returns.contains("value") && returns["value"].is_null())
                return "NoneType";
              else if (returns.contains("value"))
                return "Any"; // Other constant types
            }
            // Handle other annotation types
            else
            {
              // Try to extract id if it exists
              if (returns.contains("id"))
                return returns["id"];
            }
          }
          // Handle case where returns exists but doesn't have expected structure
          else
          {
            log_warning(
              "Unrecognized return type annotation for function "
              "{}",
              func_name);
            return "Any"; // Safe default
          }
        }
        // If function has no explicit return statements, assume void/None
        return "NoneType";
      }
    }

    // Check for lambda assignments when regular function not found
    Json lambda_elem = find_lambda_in_body(func_name, ast["body"]);

    // Also check current function scope if not found globally
    if (lambda_elem.empty() && current_func != nullptr)
      lambda_elem = find_lambda_in_body(func_name, (*current_func)["body"]);

    if (!lambda_elem.empty())
      return infer_lambda_return_type(lambda_elem);

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

    // Check if the function is a built-in function
    auto it = builtin_functions.find(func_name);
    if (it != builtin_functions.end())
    {
      return it->second;
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

    if (
      list["_type"] == "Call" && list.contains("func") &&
      list["func"].contains("attr") && list["func"]["attr"] == "array")
      return get_list_subtype(list["args"][0]);

    if (!list.contains("elts"))
      return "";

    if (!list["elts"].empty())
      list_subtype = get_type_from_json(list["elts"][0]["value"]);

    for (const auto &elem : list["elts"])
      if (get_type_from_json(elem["value"]) != list_subtype)
        throw std::runtime_error("Multiple typed lists are not supported\n");

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
        return type;
    }

    // Find RHS variable declaration in the current function
    if (rhs_node.empty() && current_func)
      rhs_node = find_annotated_assign(rhs_var_name, (*current_func)["body"]);

    // Check current function scope for function parameters
    if (
      rhs_node.empty() && current_func != nullptr &&
      (*current_func).contains("args"))
    {
      rhs_node =
        find_annotated_assign(rhs_var_name, (*current_func)["args"]["args"]);
    }

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
      // Check if annotation exists and is not null before accessing
      if (
        rhs_node.contains("annotation") && !rhs_node["annotation"].is_null() &&
        rhs_node["annotation"].contains("id"))
      {
        // Get the type of the variable being subscripted
        const std::string &var_type = rhs_node["annotation"]["id"];

        // Handle string subscript: str[index] returns str
        if (var_type == "str")
          return "str";
        // Handle list subscript: use existing logic
        else if (var_type == "list")
        {
          if (!rhs_node["value"].is_null())
            return get_list_subtype(rhs_node["value"]);
          else
            return "Any"; // Default for unknown list element types
        }
        // Handle other subscript operations
        else
          return "Any"; // Default for unknown subscript types
      }
      else
        return "Any"; // Default for unknown subscript types when annotation is missing
    }

    if (
      rhs_node.contains("annotation") && !rhs_node["annotation"].is_null() &&
      rhs_node["annotation"].contains("id") &&
      !rhs_node["annotation"]["id"].is_null())
    {
      return rhs_node["annotation"]["id"];
    }
    else
      return "Any"; // Default for cases where annotation is missing or null
  }

  std::string get_type_from_call(const Json &element)
  {
    const std::string &func_id = element["value"]["func"]["id"];

    if (
      json_utils::is_class<Json>(func_id, ast_) ||
      type_utils::is_builtin_type(func_id) ||
      type_utils::is_consensus_type(func_id) ||
      type_utils::is_python_exceptions(func_id))
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

    std::string obj_type;
    if (
      obj_node.contains("annotation") && !obj_node["annotation"].is_null() &&
      obj_node["annotation"].contains("id") &&
      !obj_node["annotation"]["id"].is_null())
    {
      obj_type = obj_node["annotation"]["id"].template get<std::string>();
    }
    else
      obj_type = "Any"; // Default fallback type

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
      if (!stmt.contains("annotation") || stmt["annotation"].is_null())
        return InferResult::UNKNOWN;

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
      inferred_type = get_type_from_constant(stmt["value"]);
    else if (value_type == "List")
      inferred_type = "list";
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
      inferred_type = get_type_from_rhs_variable(stmt, body);

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

      if (stmt_type == "If" || stmt_type == "While" || stmt_type == "Try")
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

      const auto &rhs = element["value"];

      if (
        (rhs["_type"] == "Constant" && rhs["value"].is_null()) ||
        (rhs["_type"] == "NameConstant" && rhs["value"].is_null()) ||
        (rhs["_type"] == "Name" && rhs.contains("id") && rhs["id"] == "None"))
        inferred_type = "NoneType";

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
      id = target["id"];
    // Get LHS from members access on assignments. e.g.: x.data = 10
    else if (target["_type"] == "Attribute")
    {
      id = target["value"]["id"].template get<std::string>() + "." +
           target["attr"].template get<std::string>();
    }
    else if (target.contains("slice"))
      return; // No need to annotate assignments to array elements.

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
        update_offsets(arg);
    }

    // Adjust column offsets in function call node
    if (element["value"].contains("func"))
      update_offsets(element["value"]["func"]);
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
