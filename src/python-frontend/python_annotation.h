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
    : ast_(ast),
      gs_(gs),
      current_func(nullptr),
      parent_func(nullptr),
      current_line_(0)
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
  // Infer return type from non-recursive return statements
  std::string
  infer_from_return_statements(const Json &body, const std::string &func_name)
  {
    for (const Json &stmt : body)
    {
      // Found a return statement
      if (stmt["_type"] == "Return" && !stmt["value"].is_null())
      {
        const Json &return_val = stmt["value"];

        // Skip recursive calls
        if (
          return_val["_type"] == "Call" && return_val.contains("func") &&
          return_val["func"]["_type"] == "Name" &&
          return_val["func"]["id"] == func_name)
        {
          continue; // Skip this recursive call
        }

        // Handle function calls (including nested functions)
        if (
          return_val["_type"] == "Call" && return_val.contains("func") &&
          return_val["func"]["_type"] == "Name")
        {
          const std::string &called_func = return_val["func"]["id"];

          // Try to get the return type of the called function
          try
          {
            std::string called_func_type =
              get_function_return_type(called_func, ast_);
            if (!called_func_type.empty() && called_func_type != "NoneType")
              return called_func_type;
          }
          catch (std::runtime_error &)
          {
            // Function not found, continue with normal inference
          }
        }

        // Reuse get_argument_type to infer the return value type
        std::string inferred_type = get_argument_type(return_val);
        if (!inferred_type.empty())
          return inferred_type;
      }

      // Recursively check nested blocks (if/while/for/try)
      if (stmt.contains("body") && stmt["body"].is_array())
      {
        std::string nested_type =
          infer_from_return_statements(stmt["body"], func_name);
        if (!nested_type.empty())
          return nested_type;
      }

      if (stmt.contains("orelse") && stmt["orelse"].is_array())
      {
        std::string nested_type =
          infer_from_return_statements(stmt["orelse"], func_name);
        if (!nested_type.empty())
          return nested_type;
      }
    }

    return ""; // Couldn't infer
  }

  // Method to infer and annotate unannotated function parameters
  void infer_parameter_types(Json &function_element)
  {
    const std::string &func_name = function_element["name"];

    // Determine where to search for function calls:
    // - For nested functions: search in parent function's body
    // - For top-level functions: search in global AST body
    const Json &search_context =
      (parent_func != nullptr)
        ? (*parent_func)["body"] // Search in parent function's body
        : ast_["body"];          // Search in global body

    // Find all calls to this function in the AST
    std::vector<Json> function_calls =
      find_function_calls(func_name, search_context);

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
      const std::string &var_name = arg["id"];

      // Try to find the type of the variable in current scope
      Json var_node =
        json_utils::find_var_decl(var_name, get_current_func_name(), ast_);

      // Check if we found a node with annotation
      bool has_annotation = !var_node.empty() &&
                            var_node.contains("annotation") &&
                            !var_node["annotation"].is_null();

      // If not found or found without annotation, and we're in a nested function,
      // try to find in parent function's parameters and body
      if (!has_annotation && parent_func != nullptr)
      {
        // Check parent function's parameters
        if (
          (*parent_func).contains("args") &&
          (*parent_func)["args"].contains("args"))
        {
          var_node =
            find_annotated_assign(var_name, (*parent_func)["args"]["args"]);
        }

        // If still not found, check parent function's body for local variables
        if (var_node.empty() && (*parent_func).contains("body"))
        {
          var_node = find_annotated_assign(var_name, (*parent_func)["body"]);
        }
      }

      // Extract type from the found variable node
      if (
        !var_node.empty() && var_node.contains("annotation") &&
        !var_node["annotation"].is_null())
      {
        // Handle arg nodes (function parameters)
        if (var_node["_type"] == "arg")
        {
          // Parameters can have annotation as {"id": "int"} or
          // {"value": {"id": "int"}} for generic types
          if (var_node["annotation"].contains("id"))
            return var_node["annotation"]["id"];
          else if (
            var_node["annotation"].contains("value") &&
            var_node["annotation"]["value"].contains("id"))
            return var_node["annotation"]["value"]["id"];
        }
        // Handle generic type annotations like list[int] (Subscript nodes)
        else if (
          var_node["annotation"].contains("_type") &&
          var_node["annotation"]["_type"] == "Subscript" &&
          var_node["annotation"].contains("value") &&
          var_node["annotation"]["value"].contains("id"))
        {
          std::string base_type = var_node["annotation"]["value"]["id"];
          return base_type; // Return base type if slice info unavailable
        }
        // Handle simple type annotations like int, str (Name nodes)
        else if (var_node["annotation"].contains("id"))
        {
          return var_node["annotation"]["id"];
        }
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
    else if (arg["_type"] == "Call")
    {
      // Handle function calls like abs(a - b), len(list), etc.
      if (arg["func"]["_type"] == "Name")
      {
        const std::string &func_name = arg["func"]["id"];

        // Check built-in functions first
        auto it = builtin_functions.find(func_name);
        if (it != builtin_functions.end())
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
  std::string get_list_type_from_literal(const Json &list_arg)
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
            auto func_node = json_utils::find_function(ast_["body"], func_name);
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
    std::string saved_func_name_context = current_func_name_context_;
    Json *saved_parent_func = parent_func; // Save previous parent

    const std::string &func_name =
      function_element["name"].template get<std::string>();

    // Build hierarchical path ONLY if we're not inside a class
    if (!current_class_name_.empty())
    {
      // We're inside a class - do NOT accumulate hierarchical context
      current_func_name_context_ = func_name;
    }
    else if (!saved_func_name_context.empty())
    {
      // Nested function outside a class - accumulate context
      current_func_name_context_ = saved_func_name_context + "@F@" + func_name;
    }
    else
    {
      // Top-level function
      current_func_name_context_ = func_name;
    }

    parent_func = current_func; // Current becomes parent
    current_func = &function_element;

    // Infer types for unannotated parameters based on function calls
    infer_parameter_types(function_element);

    // Add type annotations within the function
    add_annotation(function_element);

    // Skip return type inference for __init__ (constructors always return None)
    if (func_name == "__init__")
    {
      // Constructors return None by default - no need to infer
      if (function_element["returns"].is_null())
      {
        function_element["returns"] = {
          {"_type", "Constant"},
          {"value", nullptr}, // None
          {"lineno", function_element["lineno"]},
          {"col_offset", function_element["col_offset"]},
          {"end_lineno", function_element["lineno"]},
          {"end_col_offset",
           function_element["col_offset"].template get<int>() + 4}};
      }

      // Update the end column offset after adding annotations
      update_end_col_offset(function_element);
      current_func = nullptr;
      parent_func = saved_parent_func; // Restore previous parent
      current_func_name_context_ = saved_func_name_context;
      return; // Exit early for __init__
    }

    // Check if we should override the return annotation
    bool should_override =
      config.options.get_bool_option("override-return-annotation");
    bool has_no_annotation = function_element["returns"].is_null();

    if (has_no_annotation || should_override)
    {
      std::string inferred_type =
        infer_from_return_statements(function_element["body"], func_name);

      // Only add annotation if we successfully inferred a type from return statements
      if (!inferred_type.empty())
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
      // If no return type could be inferred, leave returns as null
      // (function has no explicit return statement)
    }

    // Update the end column offset after adding annotations
    update_end_col_offset(function_element);

    current_func = nullptr;
    parent_func = saved_parent_func; // Restore previous parent
    current_func_name_context_ = saved_func_name_context;
  }

  void annotate_class(Json &class_element)
  {
    std::string saved_class_name = current_class_name_;
    std::string saved_context = current_func_name_context_;
    //    current_func_name_context_ = ""; // Reset for class methods

    current_class_name_ = class_element["name"].template get<std::string>();

    for (Json &class_member : class_element["body"])
    {
      // Process methods in the class
      if (class_member["_type"] == "FunctionDef")
      {
        // Add type annotations within the class member function
        annotate_function(class_member);
      }
      // Process unannotated class attributes (e.g., species = "Homo sapiens")
      else if (class_member["_type"] == "Assign")
      {
        std::string inferred_type;

        // Infer type from the RHS value
        if (
          infer_type(class_member, class_element, inferred_type) ==
          InferResult::OK)
        {
          // Convert Assign to AnnAssign with the inferred type
          update_assignment_node(class_member, inferred_type);
        }
        else
        {
          // If type inference fails, throw error with helpful message
          std::string attr_name =
            class_member["targets"][0]["id"].template get<std::string>();
          throw std::runtime_error(
            "Cannot infer type for class attribute '" + attr_name +
            "' in class '" + class_element["name"].template get<std::string>() +
            "'. Please add explicit type annotation.");
        }
      }
    }

    current_class_name_ = saved_class_name;
    current_func_name_context_ = saved_context;
  }

  std::string get_current_func_name()
  {
    return current_func_name_context_;
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

  // Helper method to recursively search for a function in the AST
  Json
  find_function_recursive(const std::string &func_name, const Json &body) const
  {
    for (const Json &elem : body)
    {
      // Found the function at this level
      if (elem["_type"] == "FunctionDef" && elem["name"] == func_name)
      {
        return elem;
      }

      // Recursively search in nested function bodies
      if (elem["_type"] == "FunctionDef" && elem.contains("body"))
      {
        Json nested = find_function_recursive(func_name, elem["body"]);
        if (!nested.empty())
          return nested;
      }

      // Also search in control flow blocks
      if (elem.contains("body") && elem["body"].is_array())
      {
        Json nested = find_function_recursive(func_name, elem["body"]);
        if (!nested.empty())
          return nested;
      }

      if (elem.contains("orelse") && elem["orelse"].is_array())
      {
        Json nested = find_function_recursive(func_name, elem["orelse"]);
        if (!nested.empty())
          return nested;
      }
    }

    return Json(); // Not found
  }

  std::string
  get_function_return_type(const std::string &func_name, const Json &ast)
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
              else if (returns.contains("value"))
                return "Any"; // Other constant types
            }
            // Handle other annotation types
            else
            {
              // Try to extract id if it exists
              if (returns.contains("id"))
              {
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
      auto return_node = json_utils::find_return_node(func_elem["body"]);
      if (!return_node.empty())
      {
        std::string inferred_type;
        infer_type(return_node, func_elem, inferred_type);
        functions_in_analysis_.erase(func_name);
        return inferred_type;
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

    // Check if the function is a built-in function
    auto it = builtin_functions.find(func_name);
    if (it != builtin_functions.end())
    {
      functions_in_analysis_.erase(func_name);
      return it->second;
    }

    functions_in_analysis_.erase(func_name);

    std::ostringstream oss;
    oss << "Function \"" << func_name << "\" not found (" << python_filename_
        << " line " << current_line_ << ")";

    throw std::runtime_error(oss.str());
  }

  std::string get_type_from_lhs(const std::string &id, Json &body)
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

  std::string get_base_var_name(const Json &node) const
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

  bool extract_type_info(
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

  std::string infer_dict_value_type(const Json &var_node)
  {
    if (!var_node.contains("value") || var_node["value"].is_null())
      return "Any";

    const Json &dict_init = var_node["value"];

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

  std::string
  resolve_subscript_type(const Json &subscript_node, const Json &body)
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

    return base_type;
  }

  std::string get_type_from_rhs_variable(const Json &element, const Json &body)
  {
    const auto &value_type = element["value"]["_type"];

    // Handle subscript of string constant (e.g., "hello"[0] returns str)
    if (
      value_type == "Subscript" &&
      element["value"]["value"]["_type"] == "Constant" &&
      element["value"]["value"]["value"].is_string())
    {
      return "str";
    }

    // Handle subscript operations (including nested ones)
    if (value_type == "Subscript")
      return resolve_subscript_type(element["value"], body);

    std::string rhs_var_name;

    if (value_type == "Name")
      rhs_var_name = element["value"]["id"];
    else if (value_type == "UnaryOp")
      rhs_var_name = element["value"]["operand"]["id"];
    else
      rhs_var_name = get_base_var_name(element["value"]["value"]);

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

  std::string match_literal_argument(
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
  std::string resolve_overload_return_type(
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

  std::string get_type_from_call(const Json &element)
  {
    const Json &func = element["value"]["func"];

    // Handle regular function calls like func_name()
    if (func["_type"] == "Name")
    {
      const std::string &func_id = func["id"];

      if (
        json_utils::is_class<Json>(func_id, ast_) ||
        type_utils::is_builtin_type(func_id) ||
        type_utils::is_consensus_type(func_id) ||
        type_utils::is_python_exceptions(func_id))
        return func_id;

      if (type_utils::is_consensus_func(func_id))
        return type_utils::get_type_from_consensus_func(func_id);

      // Try to resolve overload before falling back to regular function
      if (!type_utils::is_python_model_func(func_id))
      {
        std::string overload_type =
          resolve_overload_return_type(func_id, element["value"]);
        if (!overload_type.empty())
          return overload_type;

        return get_function_return_type(func_id, ast_);
      }
    }

    // Handle class method calls like int.from_bytes(), str.join(), etc.
    else if (func["_type"] == "Attribute" && func["value"]["_type"] == "Name")
    {
      const std::string &class_name = func["value"]["id"];
      const std::string &method_name = func["attr"];

      // Handle built-in type class methods
      if (type_utils::is_builtin_type(class_name))
      {
        // Map specific class methods to their return types
        if (class_name == "int" && method_name == "from_bytes")
          return "int";
        else if (
          class_name == "str" &&
          (method_name == "join" || method_name == "format"))
          return "str";
        else if (
          class_name == "bytes" &&
          (method_name == "decode" || method_name == "hex"))
          return "str";
        else if (class_name == "bytes" && method_name == "fromhex")
          return "bytes";
        else if (class_name == "list" && method_name == "copy")
          return "list";
        else if (
          class_name == "dict" &&
          (method_name == "copy" || method_name == "fromkeys"))
          return "dict";
        else
          return class_name; // Default: method returns same type as class
      }

      if (module_manager_)
      {
        auto module = module_manager_->get_module(class_name);
        if (module)
        {
          auto overloads_funcs = module->overloads();
          std::vector<Json> overloads;

          for (const auto &elem : overloads_funcs)
          {
            if (elem["_type"] == "FunctionDef" && elem["name"] == method_name)
              overloads.push_back(elem);
          }

          return match_literal_argument(element["value"], overloads);
        }
      }
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

    // Handle method calls on constant literals
    // Python allows: " ".join(l), 123.to_bytes(), etc.
    // Before this fix, accessing call["value"]["id"] would crash because
    // Constant nodes don't have an "id" field
    if (call["value"]["_type"] == "Constant")
      return get_type_from_constant(call["value"]);

    // Handle normal Name values (variable references)
    if (!call["value"].contains("id"))
      return "";

    obj_name += call["value"]["id"].template get<std::string>();
    if (obj_name.find('.') != std::string::npos)
      obj_name = invert_substrings(obj_name);

    return json_utils::get_object_alias(ast_, obj_name);
  }

  std::string get_type_from_method(const Json &call)
  {
    std::string type("");

    // Handle method calls on constant literals
    // When Python code has " ".join(l), the func["value"] is a Constant node
    // We need to map string method names to their return types directly
    // without looking up the object in the AST (which would fail)
    if (
      call["func"].contains("value") &&
      call["func"]["value"]["_type"] == "Constant")
    {
      std::string obj_type = get_type_from_constant(call["func"]["value"]);

      // For string constants, determine return type based on method name
      if (obj_type == "str" && call["func"].contains("attr"))
      {
        const std::string &method = call["func"]["attr"];
        // Methods that return str
        if (
          method == "join" || method == "lower" || method == "upper" ||
          method == "strip" || method == "lstrip" || method == "rstrip" ||
          method == "format" || method == "replace")
          return "str";
        // Methods that return bool
        else if (
          method == "startswith" || method == "endswith" ||
          method == "isdigit" || method == "isalpha" || method == "isspace" ||
          method == "islower" || method == "isupper")
          return "bool";
        // Default for string methods
        return "str";
      }

      return obj_type;
    }

    const std::string &obj = get_object_name(call["func"], std::string());
    std::string attr_name = call["func"].contains("attr")
                              ? call["func"]["attr"].template get<std::string>()
                              : "";

    // Get type from imported module
    if (module_manager_)
    {
      // Try to get module using the object name directly
      auto module = module_manager_->get_module(obj);

      // If not found, try using get_object_alias to resolve import aliases
      if (!module && !obj.empty())
      {
        std::string resolved_obj = json_utils::get_object_alias(ast_, obj);
        if (resolved_obj != obj)
        {
          module = module_manager_->get_module(resolved_obj);
        }
      }

      if (module)
      {
        // First try as function
        function func = module->get_function(attr_name);
        if (!func.name_.empty())
        {
          return func.return_type_;
        }

        // If not a function, try as class
        class_definition cls = module->get_class(attr_name);
        if (!cls.name_.empty())
        {
          // Return the class name as the type for constructor calls
          return cls.name_;
        }

        // If module exists but attribute not found, don't continue to search
        // in AST (which would fail for imported modules)
        return "";
      }
    }

    Json obj_node =
      json_utils::find_var_decl(obj, get_current_func_name(), ast_);

    // Check current function parameters if not found
    if (
      obj_node.empty() && current_func != nullptr &&
      (*current_func).contains("args") &&
      (*current_func)["args"].contains("args"))
    {
      obj_node = find_annotated_assign(obj, (*current_func)["args"]["args"]);
    }

    // Handle nested attribute access (e.g., self.f.foo() or self.b.a.get_value())
    if (obj_node.empty() && call["func"]["value"]["_type"] == "Attribute")
    {
      // Recursively resolve nested attribute chain (e.g., self.b.a -> [self, b, a])
      std::function<std::string(const Json &, std::vector<std::string> &)>
        extract_attr_chain =
          [&](
            const Json &node, std::vector<std::string> &chain) -> std::string {
        if (node["_type"] == "Attribute")
        {
          std::string attr = node["attr"].template get<std::string>();
          chain.push_back(attr);
          return extract_attr_chain(node["value"], chain);
        }
        else if (node["_type"] == "Name" && node.contains("id"))
        {
          return node["id"].template get<std::string>();
        }
        return "";
      };

      std::vector<std::string> attr_chain;
      std::string base_obj =
        extract_attr_chain(call["func"]["value"], attr_chain);

      // Reverse the chain because extract_attr_chain collects from outer to inner
      // For self.b.a, we get [a, b] but need [b, a] to process left to right
      std::reverse(attr_chain.begin(), attr_chain.end());

      // If base object is "self" and we have at least one attribute, resolve the chain
      if (base_obj == "self" && !attr_chain.empty() && current_func != nullptr)
      {
        // Get the class name from current_func's context
        std::string class_name = "";
        for (const Json &elem : ast_["body"])
        {
          if (elem["_type"] == "ClassDef" && elem.contains("body"))
          {
            for (const Json &member : elem["body"])
            {
              if (
                member["_type"] == "FunctionDef" &&
                member["name"] == (*current_func)["name"])
              {
                class_name = elem["name"].template get<std::string>();
                break;
              }
            }
            if (!class_name.empty())
              break;
          }
        }

        // Recursively resolve the attribute chain
        std::string current_type = class_name;
        for (size_t i = 0; i < attr_chain.size(); ++i)
        {
          const std::string &attr_name = attr_chain[i];

          // Find the current class
          Json current_class =
            json_utils::find_class(ast_["body"], current_type);
          if (current_class.empty())
            break;

          // Look for the attribute in __init__ method
          bool found = false;
          for (const Json &member : current_class["body"])
          {
            if (
              member["_type"] == "FunctionDef" && member["name"] == "__init__")
            {
              for (const Json &stmt : member["body"])
              {
                // Check for AnnAssign: self.attr: Type = value
                if (
                  stmt["_type"] == "AnnAssign" &&
                  stmt["target"]["_type"] == "Attribute" &&
                  stmt["target"]["value"]["id"] == "self" &&
                  stmt["target"]["attr"] == attr_name)
                {
                  // Found the attribute, get its type
                  if (
                    stmt.contains("annotation") &&
                    !stmt["annotation"].is_null() &&
                    stmt["annotation"].contains("id") &&
                    !stmt["annotation"]["id"].is_null())
                  {
                    current_type =
                      stmt["annotation"]["id"].template get<std::string>();
                    found = true;
                    break;
                  }
                }
              }
              if (found)
                break;
            }
          }

          if (!found)
            break;

          // If this is the last attribute in the chain, find the method return type
          if (i == attr_chain.size() - 1)
          {
            Json final_class =
              json_utils::find_class(ast_["body"], current_type);
            if (!final_class.empty())
            {
              const std::string &method_name = call["func"]["attr"];
              for (const Json &method : final_class["body"])
              {
                if (
                  method["_type"] == "FunctionDef" &&
                  method["name"] == method_name)
                {
                  if (
                    method.contains("returns") &&
                    method["returns"].contains("id"))
                  {
                    return method["returns"]["id"].template get<std::string>();
                  }
                }
              }
            }
          }
        }
      }
    }

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

    // Handle built-in types
    if (type_utils::is_builtin_type(obj_type))
      type = obj_type;
    else
    {
      // Handle user-defined class methods
      Json class_node = json_utils::find_class(ast_["body"], obj_type);
      if (!class_node.empty())
      {
        const std::string &method_name = call["func"]["attr"];

        // Find the method in the class body
        for (const Json &member : class_node["body"])
        {
          if (member["_type"] == "FunctionDef" && member["name"] == method_name)
          {
            // Get return type from annotation
            if (
              member.contains("returns") && !member["returns"].is_null() &&
              member["returns"].contains("id"))
            {
              return member["returns"]["id"];
            }
          }
        }
      }
    }

    return type;
  }

  // Method to infer type from conditional expressions (IfExp)
  std::string get_type_from_ifexp(const Json &ifexp_node, const Json &body)
  {
    // For conditional expressions like "x + y if x > y else x * y"
    // We need to infer the type from both the body and orelse branches

    // Create temporary statement nodes for type inference
    Json body_stmt = {
      {"_type", "Assign"},
      {"value", ifexp_node["body"]},
      {"lineno", current_line_}};

    Json orelse_stmt = {
      {"_type", "Assign"},
      {"value", ifexp_node["orelse"]},
      {"lineno", current_line_}};

    std::string body_type, orelse_type;

    // Infer type from the body (true branch)
    InferResult body_result = infer_type(body_stmt, body, body_type);

    // Infer type from the orelse (false branch)
    InferResult orelse_result = infer_type(orelse_stmt, body, orelse_type);

    // If both branches have the same type, return that type
    if (body_result == InferResult::OK && orelse_result == InferResult::OK)
    {
      if (body_type == orelse_type)
        return body_type;

      // Handle numeric type promotion: if one is int and other is float, result is float
      if (
        (body_type == "int" && orelse_type == "float") ||
        (body_type == "float" && orelse_type == "int"))
        return "float";

      // For different types, use a safe fallback
      log_warning(
        "Conditional expression has different types in branches: {} vs {}. "
        "Using 'Any' as fallback ({}:{})",
        body_type,
        orelse_type,
        python_filename_,
        current_line_);
      return "Any";
    }

    // If only one branch succeeded, use that type
    if (body_result == InferResult::OK)
      return body_type;
    if (orelse_result == InferResult::OK)
      return orelse_type;

    // If neither branch could be inferred, return empty string
    return "";
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
      else if (stmt["annotation"].contains("id"))
        inferred_type = stmt["annotation"]["id"].template get<std::string>();
      else if (
        stmt["annotation"].contains("_type") &&
        stmt["annotation"]["_type"] == "BinOp")
      {
        // Handle union types such as str | None (PEP 604 syntax)
        const auto &left = stmt["annotation"]["left"];
        const auto &right = stmt["annotation"]["right"];

        // Check which side is None and extract the other type
        bool left_is_none =
          (left.contains("_type") && left["_type"] == "Constant" &&
           left.contains("value") && left["value"].is_null());
        bool right_is_none =
          (right.contains("_type") && right["_type"] == "Constant" &&
           right.contains("value") && right["value"].is_null());

        if (right_is_none && left.contains("id"))
          inferred_type = left["id"].template get<std::string>();
        else if (left_is_none && right.contains("id"))
          inferred_type = right["id"].template get<std::string>();
        else
          return InferResult::UNKNOWN;
      }
      else
        return InferResult::UNKNOWN;

      return InferResult::OK;
    }

    const auto &value_type = stmt["value"]["_type"];

    // Get type from RHS constant
    if (value_type == "Constant")
      inferred_type = get_type_from_constant(stmt["value"]);
    else if (value_type == "List")
      inferred_type = "list";
    else if (value_type == "Set")
      inferred_type = "set";
    else if (value_type == "Tuple")
      inferred_type = "tuple";
    else if (value_type == "Dict")
      inferred_type = "dict";
    else if (value_type == "Compare")
      inferred_type = "bool";
    else if (value_type == "UnaryOp") // Handle negative numbers
    {
      const auto &operand = stmt["value"]["operand"];
      const auto &operand_type = operand["_type"];
      if (operand_type == "Constant")
        inferred_type = get_type_from_constant(operand);
      else if (operand_type == "Name")
        inferred_type = get_type_from_rhs_variable(stmt, body);
      else if (operand_type == "BinOp")
      {
        // Handle unary operations on binary expressions like -a ** b
        Json temp_stmt = {{"value", operand}};
        inferred_type = get_type_from_binary_expr(temp_stmt, body);
      }
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

    // Get type from conditional expressions (ternary operator)
    else if (value_type == "IfExp")
      inferred_type = get_type_from_ifexp(stmt["value"], body);

    // Get type from top-level functions
    else if (
      value_type == "Call" && stmt["value"]["func"]["_type"] == "Name" &&
      !type_utils::is_python_model_func(stmt["value"]["func"]["id"]))
    {
      inferred_type = get_type_from_call(stmt);
    }

    // Get type from methods and class methods
    else if (
      value_type == "Call" && stmt["value"]["func"]["_type"] == "Attribute")
    {
      // Try get_type_from_call first (checks builtin_functions map)
      inferred_type = get_type_from_call(stmt);

      // If that didn't work, try get_type_from_method
      if (inferred_type.empty())
        inferred_type = get_type_from_method(stmt["value"]);

      // Handle module.Class() constructor calls (e.g., datetime.datetime(...))
      // Use the attribute name as the type if nothing else worked
      if (inferred_type.empty())
      {
        const auto &func = stmt["value"]["func"];
        if (
          func.contains("attr") && func.contains("value") &&
          func["value"]["_type"] == "Name" && func["value"].contains("id"))
        {
          std::string attr_name = func["attr"].template get<std::string>();

          // Only use attribute as type if it looks like a constructor call
          // (i.e., attribute name starts with uppercase)
          if (!attr_name.empty() && std::isupper(attr_name[0]))
            inferred_type = attr_name;
          // For method calls that couldn't be resolved, return UNKNOWN
          else
            return InferResult::UNKNOWN;
        }
      }
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

        // Process else block if it exists
        if (
          stmt_type == "If" && element.contains("orelse") &&
          !element["orelse"].empty())
        {
          // Create a temporary body structure for the else block
          Json else_body = {{"body", element["orelse"]}};
          add_annotation(else_body);
          // Update the original orelse with annotated version
          element["orelse"] = else_body["body"];
        }

        continue;
      }

      if (stmt_type == "FunctionDef")
      {
        // Only annotate nested functions, not the current function itself
        if (current_func != nullptr && &element != current_func)
          annotate_function(element);

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

      // Skip tuple/list unpacking assignments
      // The C++ converter will handle them directly with proper type inference
      if (
        element.contains("targets") && !element["targets"].empty() &&
        element["targets"][0].contains("_type") &&
        (element["targets"][0]["_type"] == "Tuple" ||
         element["targets"][0]["_type"] == "List"))
      {
        continue;
      }

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

  Json create_name_annotation(
    const std::string &type_id,
    int lineno,
    int col_offset,
    int end_lineno,
    int end_col_offset)
  {
    return {
      {"_type", "Name"},
      {"id", type_id},
      {"ctx", {{"_type", "Load"}}},
      {"lineno", lineno},
      {"col_offset", col_offset},
      {"end_lineno", end_lineno},
      {"end_col_offset", end_col_offset}};
  }

  Json create_subscript_annotation(
    const std::string &base_type,
    const std::string &element_type,
    int lineno,
    int col_offset,
    int end_lineno)
  {
    int base_end_col = col_offset + base_type.size();
    int slice_col = base_end_col + 1; // After '['
    int slice_end_col = slice_col + element_type.size();
    int total_end_col = col_offset + base_type.size() + 1 +
                        element_type.size() + 1; // type[element]

    return {
      {"_type", "Subscript"},
      {"value",
       create_name_annotation(
         base_type, lineno, col_offset, end_lineno, base_end_col)},
      {"slice",
       create_name_annotation(
         element_type, lineno, slice_col, end_lineno, slice_end_col)},
      {"ctx", {{"_type", "Load"}}},
      {"lineno", lineno},
      {"col_offset", col_offset},
      {"end_lineno", end_lineno},
      {"end_col_offset", total_end_col}};
  }

  Json create_annotation_from_type(
    const std::string &inferred_type,
    int lineno,
    int col_offset,
    int end_lineno)
  {
    // Check if this is a generic type (e.g., list[dict])
    size_t bracket_pos = inferred_type.find('[');

    if (bracket_pos != std::string::npos)
    {
      // Generic type: extract base and element types
      std::string base_type = inferred_type.substr(0, bracket_pos);
      std::string element_type = inferred_type.substr(
        bracket_pos + 1, inferred_type.length() - bracket_pos - 2);

      return create_subscript_annotation(
        base_type, element_type, lineno, col_offset, end_lineno);
    }
    else
    {
      // Simple type
      int end_col_offset = col_offset + inferred_type.size();
      return create_name_annotation(
        inferred_type, lineno, col_offset, end_lineno, end_col_offset);
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

    // Calculate column offset with null safety
    int target_col_offset =
      target.contains("col_offset") && !target["col_offset"].is_null()
        ? target["col_offset"].template get<int>()
        : 0;
    int col_offset = target_col_offset + id.size() + 1;

    // Get line number with null safety
    int target_lineno = target.contains("lineno") && !target["lineno"].is_null()
                          ? target["lineno"].template get<int>()
                          : current_line_;

    int target_end_lineno =
      target.contains("end_lineno") && !target["end_lineno"].is_null()
        ? target["end_lineno"].template get<int>()
        : target_lineno;

    element["annotation"] = create_annotation_from_type(
      inferred_type, target_lineno, col_offset, target_end_lineno);

    // Update element properties with null safety
    int element_end_col_offset =
      element.contains("end_col_offset") && !element["end_col_offset"].is_null()
        ? element["end_col_offset"].template get<int>()
        : col_offset + inferred_type.size();

    element["end_col_offset"] =
      element_end_col_offset + inferred_type.size() + 1;
    element["end_lineno"] = target_lineno;
    element["simple"] = 1;

    // Replace "targets" array with "target" object
    element["target"] = std::move(target);
    element.erase("targets");

    // Remove unnecessary field
    element.erase("type_comment");

    // Update value fields with the correct offsets - with null safety
    auto update_offsets = [&inferred_type](Json &value) {
      if (value.contains("col_offset") && !value["col_offset"].is_null())
      {
        value["col_offset"] =
          value["col_offset"].template get<int>() + inferred_type.size() + 1;
      }
      if (
        value.contains("end_col_offset") && !value["end_col_offset"].is_null())
      {
        value["end_col_offset"] = value["end_col_offset"].template get<int>() +
                                  inferred_type.size() + 1;
      }
    };

    update_offsets(element["value"]);

    // Adjust column offsets for function calls with arguments
    if (element["value"].contains("args"))
      for (auto &arg : element["value"]["args"])
        update_offsets(arg);

    // Adjust column offsets in function call node
    if (element["value"].contains("func"))
      update_offsets(element["value"]["func"]);
  }

  void add_parameter_annotation(Json &param, const std::string &type)
  {
    int col_offset = param["col_offset"].template get<int>() +
                     param["arg"].template get<std::string>().length() + 1;

    param["annotation"] = create_annotation_from_type(
      type,
      param["lineno"].template get<int>(),
      col_offset,
      param["lineno"].template get<int>());

    // Update the parameter's end_col_offset to account for the annotation
    param["end_col_offset"] =
      param["end_col_offset"].template get<int>() + type.size() + 1;
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
      // Check if this element is the variable we're looking for
      if (
        elem.contains("_type") &&
        ((elem["_type"] == "AnnAssign" && elem.contains("target") &&
          elem["target"].contains("id") &&
          elem["target"]["id"].template get<std::string>() == node_name) ||
         (elem["_type"] == "arg" && elem["arg"] == node_name)))
      {
        return elem;
      }

      // Recursively search inside nested blocks
      if (elem.contains("_type"))
      {
        const std::string &elem_type = elem["_type"];

        // Search inside If blocks (both if and else branches)
        if (elem_type == "If")
        {
          if (elem.contains("body") && !elem["body"].empty())
          {
            Json result = find_annotated_assign(node_name, elem["body"]);
            if (!result.empty())
              return result;
          }
          if (elem.contains("orelse") && !elem["orelse"].empty())
          {
            Json result = find_annotated_assign(node_name, elem["orelse"]);
            if (!result.empty())
              return result;
          }
        }
        // Search inside While blocks
        else if (elem_type == "While")
        {
          if (elem.contains("body") && !elem["body"].empty())
          {
            Json result = find_annotated_assign(node_name, elem["body"]);
            if (!result.empty())
              return result;
          }
        }
        // Search inside For blocks
        else if (elem_type == "For")
        {
          if (elem.contains("body") && !elem["body"].empty())
          {
            Json result = find_annotated_assign(node_name, elem["body"]);
            if (!result.empty())
              return result;
          }
        }
        // Search inside Try blocks
        else if (elem_type == "Try")
        {
          if (elem.contains("body") && !elem["body"].empty())
          {
            Json result = find_annotated_assign(node_name, elem["body"]);
            if (!result.empty())
              return result;
          }
          if (elem.contains("handlers") && !elem["handlers"].empty())
          {
            for (const Json &handler : elem["handlers"])
            {
              if (handler.contains("body") && !handler["body"].empty())
              {
                Json result = find_annotated_assign(node_name, handler["body"]);
                if (!result.empty())
                  return result;
              }
            }
          }
        }
      }
    }
    return Json();
  }

  Json &ast_;
  global_scope &gs_;
  std::shared_ptr<module_manager> module_manager_;
  Json *current_func;
  Json
    *parent_func; // Track parent function for nested function scope resolution
  int current_line_;
  std::string python_filename_;
  bool filter_global_elements_ = false;
  std::vector<Json> referenced_global_elements;
  std::set<std::string> functions_in_analysis_;
  std::string current_func_name_context_;
  std::string current_class_name_;
};
