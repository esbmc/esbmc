#include <python-frontend/function_call_builder.h>
#include <python-frontend/function_call_expr.h>
#include <python-frontend/numpy_call_expr.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/json_utils.h>
#include <python-frontend/type_utils.h>
#include <util/arith_tools.h>

#include <boost/algorithm/string/predicate.hpp>

const std::string kGetObjectSize = "__ESBMC_get_object_size";
const std::string kStrlen = "strlen";
const std::string kEsbmcAssume = "__ESBMC_assume";
const std::string kVerifierAssume = "__VERIFIER_assume";
const std::string kLoopInvariant = "__loop_invariant";
const std::string kEsbmcLoopInvariant = "__ESBMC_loop_invariant";

function_call_builder::function_call_builder(
  python_converter &converter,
  const nlohmann::json &call)
  : converter_(converter), call_(call)
{
}

bool function_call_builder::is_numpy_call(const symbol_id &function_id) const
{
  if (type_utils::is_builtin_type(function_id.get_function()))
    return false;

  const std::string &filename = function_id.get_filename();

  return boost::algorithm::ends_with(filename, "/models/numpy.py") ||
         filename.find("/numpy/linalg") != std::string::npos;
}

bool function_call_builder::is_assume_call(const symbol_id &function_id) const
{
  const std::string &func_name = function_id.get_function();
  return (func_name == kEsbmcAssume || func_name == kVerifierAssume);
}

bool function_call_builder::is_len_call(const symbol_id &function_id) const
{
  const std::string &func_name = function_id.get_function();
  return func_name == kGetObjectSize || func_name == kStrlen;
}

symbol_id function_call_builder::build_function_id() const
{
  const std::string &python_file = converter_.python_file();
  const std::string &current_class_name = converter_.current_classname();
  const std::string &current_function_name = converter_.current_function_name();
  const auto &ast = converter_.ast();
  type_handler th(converter_);

  bool is_member_function_call = false;

  const auto &func_json = call_["func"];

  const std::string &func_type = func_json["_type"];

  std::string func_name, obj_name, class_name;

  symbol_id function_id(python_file, current_class_name, current_function_name);

  if (func_type == "Name")
  {
    func_name = func_json["id"];

    // Map Python loop invariant name to ESBMC internal name
    if (func_name == kLoopInvariant)
      func_name = kEsbmcLoopInvariant;
  }
  else if (func_type == "Attribute") // Handling obj_name.func_name() calls
  {
    is_member_function_call = true;
    func_name = func_json["attr"];

    // Get object name
    if (func_json["value"]["_type"] == "Attribute")
    {
      obj_name = func_json["value"]["attr"];
    }
    else if (
      func_json["value"]["_type"] == "Constant" &&
      func_json["value"]["value"].is_string())
    {
      obj_name = "str";
    }
    else if (func_json["value"]["_type"] == "BinOp")
    {
      std::string lhs_type = th.get_operand_type(func_json["value"]["left"]);
      std::string rhs_type = th.get_operand_type(func_json["value"]["right"]);

      assert(lhs_type == rhs_type);

      obj_name = lhs_type;
    }
    else if (func_json["value"]["_type"] == "Call")
    {
      obj_name = func_json["value"]["func"]["id"];
      if (obj_name == "super")
      {
        symbolt *base_class_func = converter_.find_function_in_base_classes(
          current_class_name, function_id.to_string(), func_name, false);
        if (base_class_func)
        {
          return symbol_id::from_string(base_class_func->id.as_string());
        }
      }
    }
    else
    {
      if (
        func_json["value"]["_type"] == "Name" &&
        func_json["value"].contains("id"))
        obj_name = func_json["value"]["id"];
      else
        obj_name = "str";
    }

    obj_name = json_utils::get_object_alias(ast, obj_name);

    if (
      !json_utils::is_class(obj_name, ast) &&
      converter_.is_imported_module(obj_name))
    {
      const auto &module_path = converter_.get_imported_module_path(obj_name);

      function_id =
        symbol_id(module_path, current_class_name, current_function_name);

      is_member_function_call = false;
    }
  }

  // build symbol_id
  if (func_name == "len")
  {
    const auto &arg = call_["args"][0];
    func_name = kStrlen;

    // Special case: single character string literals should return 1 directly
    if (arg["_type"] == "Constant" && arg["value"].is_string())
    {
      std::string str_val = arg["value"].get<std::string>();
      if (str_val.size() == 1)
      {
        // For single character strings, we'll handle this specially in build()
        // by returning 1 directly instead of calling a C function
        func_name = "__ESBMC_len_single_char";
        function_id.clear();
        function_id.set_prefix("esbmc:");
        function_id.set_function(func_name);
        return function_id;
      }
    }

    if (arg["_type"] == "List")
      func_name = kGetObjectSize;
    else if (arg["_type"] == "Name")
    {
      const std::string &var_type = th.get_var_type(arg["id"]);
      // Check if this is a tuple by looking up the variable's type
      if (var_type == "tuple" || var_type.empty())
      {
        symbol_id var_sid(
          python_file, current_class_name, current_function_name);
        var_sid.set_object(arg["id"].get<std::string>());
        symbolt *var_symbol = converter_.find_symbol(var_sid.to_string());

        if (var_symbol && var_symbol->type.id() == "struct")
        {
          const struct_typet &struct_type = to_struct_type(var_symbol->type);

          // Check if this is a tuple by examining the tag
          if (struct_type.tag().as_string().find("tag-tuple") == 0)
          {
            // Mark this as a tuple len() call
            func_name = "__ESBMC_len_tuple";
            function_id.clear();
            function_id.set_prefix("esbmc:");
            function_id.set_function(func_name);
            return function_id;
          }
        }
      }
      if (
        var_type == "bytes" || var_type == "list" || var_type == "List" ||
        var_type.empty())
        func_name = kGetObjectSize;
      else if (var_type == "str")
      {
        // Check if this is a single character by looking up the variable
        symbol_id var_sid(
          python_file, current_class_name, current_function_name);
        var_sid.set_object(arg["id"].get<std::string>());
        symbolt *var_symbol = converter_.find_symbol(var_sid.to_string());

        if (
          var_symbol && var_symbol->value.is_constant() &&
          (var_symbol->value.type().is_unsignedbv() ||
           var_symbol->value.type().is_signedbv()))
        {
          // This is a single character variable
          func_name = "__ESBMC_len_single_char";
          function_id.clear();
          function_id.set_prefix("esbmc:");
          function_id.set_function(func_name);
          return function_id;
        }
      }
    }
    function_id.clear();
    function_id.set_prefix("c:");
  }
  else if (type_utils::is_builtin_type(obj_name))
  {
    class_name = obj_name;
    function_id = symbol_id(python_file, class_name, func_name);
  }
  else if (is_assume_call(function_id))
  {
    function_id.clear();
  }

  // Insert class name in the symbol id
  if (obj_name == "super")
  {
    class_name = current_class_name;
  }
  else if (th.is_constructor_call(call_))
  {
    class_name = func_name;
  }
  else if (is_member_function_call)
  {
    if (
      type_utils::is_builtin_type(obj_name) ||
      json_utils::is_class(obj_name, ast))
    {
      class_name = obj_name;
    }
    else
    {
      // Look up variable type from symbol table instead of AST
      symbol_id var_sid(python_file, current_class_name, current_function_name);
      var_sid.set_object(obj_name);
      symbolt *var_symbol = converter_.find_symbol(var_sid.to_string());

      if (!var_symbol)
        throw std::runtime_error("Variable " + obj_name + " not found");

      // Extract class name from the type, following symbol references
      typet var_type = var_symbol->type.is_pointer()
                         ? var_symbol->type.subtype()
                         : var_symbol->type;

      // Follow symbol type references using the converter's namespace
      var_type = converter_.ns.follow(var_type);

      if (var_type.is_struct())
      {
        const struct_typet &struct_type = to_struct_type(var_type);
        class_name = struct_type.tag().as_string();
      }
      else
        class_name = th.type_to_string(var_type);
    }
  }

  if (!class_name.empty())
  {
    function_id.set_class(class_name);
  }

  function_id.set_function(func_name);
  return function_id;
}

exprt function_call_builder::build() const
{
  symbol_id function_id = build_function_id();

  // Special handling for single character len() calls
  if (function_id.get_function() == "__ESBMC_len_single_char")
    return from_integer(1, int_type());

  if (function_id.get_function() == "__ESBMC_len_tuple")
  {
    const auto &arg = call_["args"][0];
    exprt obj_expr = converter_.get_expr(arg);

    if (obj_expr.type().id() == "struct")
    {
      const struct_typet &struct_type = to_struct_type(obj_expr.type());
      size_t tuple_len = struct_type.components().size();
      return from_integer(tuple_len, size_type());
    }

    // Fallback
    return from_integer(0, size_type());
  }

  // Special handling for assume calls: convert to code_assume instead of function call
  if (is_assume_call(function_id))
  {
    if (call_["args"].empty())
      throw std::runtime_error("__ESBMC_assume requires one boolean argument");

    exprt condition = converter_.get_expr(call_["args"][0]);

    // Create code_assume statement
    codet assume_code("assume");
    assume_code.copy_to_operands(condition);
    assume_code.location() = converter_.get_location_from_decl(call_);

    return assume_code;
  }

  if (call_["func"]["_type"] == "Attribute")
  {
    std::string method_name = call_["func"]["attr"].get<std::string>();

    if (method_name == "startswith")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (call_["args"].size() != 1)
        throw std::runtime_error("startswith() requires exactly one argument");

      exprt prefix_arg = converter_.get_expr(call_["args"][0]);
      locationt loc = converter_.get_location_from_decl(call_);

      return converter_.get_string_handler().handle_string_startswith(
        obj_expr, prefix_arg, loc);
    }

    if (method_name == "endswith")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (call_["args"].size() != 1)
        throw std::runtime_error("endswith() requires exactly one argument");

      exprt suffix_arg = converter_.get_expr(call_["args"][0]);
      locationt loc = converter_.get_location_from_decl(call_);

      return converter_.get_string_handler().handle_string_endswith(
        obj_expr, suffix_arg, loc);
    }

    if (method_name == "isdigit")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      if (!call_["args"].empty())
        throw std::runtime_error("isdigit() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);

      return converter_.get_string_handler().handle_string_isdigit(
        obj_expr, loc);
    }

    if (method_name == "isalpha")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);
      if (!call_["args"].empty())
        throw std::runtime_error("isalpha() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_isalpha(
        obj_expr, loc);
    }

    if (method_name == "isspace")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);
      if (!call_["args"].empty())
        throw std::runtime_error("isspace() takes no arguments");

      locationt loc = converter_.get_location_from_decl(call_);

      // Check if this is a single character (from iteration) or a string
      if (obj_expr.type().is_unsignedbv() || obj_expr.type().is_signedbv())
      {
        // Single character - use C's isspace function
        return converter_.get_string_handler().handle_char_isspace(
          obj_expr, loc);
      }
      else
      {
        // String variable - use the string version
        return converter_.get_string_handler().handle_string_isspace(
          obj_expr, loc);
      }
    }

    if (method_name == "lstrip")
    {
      exprt obj_expr = converter_.get_expr(call_["func"]["value"]);

      // lstrip() takes optional chars argument, but we only support no arguments
      if (!call_["args"].empty())
        throw std::runtime_error("lstrip() with arguments not yet supported");

      locationt loc = converter_.get_location_from_decl(call_);
      return converter_.get_string_handler().handle_string_lstrip(
        obj_expr, loc);
    }
  }

  // Add len function to symbol table
  if (is_len_call(function_id))
  {
    const auto &symbol_table = converter_.symbol_table();
    const std::string &func_symbol_id = function_id.to_string();

    if (symbol_table.find_symbol(func_symbol_id.c_str()) == nullptr)
    {
      code_typet code_type;
      code_type.return_type() = long_long_int_type();
      code_type.arguments().push_back(pointer_typet(empty_typet()));

      const std::string &python_file = converter_.python_file();
      const std::string &func_name = function_id.get_function();
      locationt location = converter_.get_location_from_decl(call_);

      symbolt symbol = converter_.create_symbol(
        python_file, func_name, func_symbol_id, location, code_type);

      converter_.add_symbol(symbol);
    }
  }

  // Add loop invariant symbol to symbol table
  if (function_id.get_function() == kEsbmcLoopInvariant)
  {
    const auto &symbol_table = converter_.symbol_table();
    const std::string &func_symbol_id = function_id.to_string();

    if (symbol_table.find_symbol(func_symbol_id.c_str()) == nullptr)
    {
      code_typet code_type;
      code_type.return_type() = empty_typet();
      code_typet::argumentt arg;
      arg.type() = bool_type();
      code_type.arguments().push_back(arg);

      const std::string &python_file = converter_.python_file();
      const std::string &func_name = function_id.get_function();
      locationt location = converter_.get_location_from_decl(call_);

      symbolt symbol = converter_.create_symbol(
        python_file, func_name, func_symbol_id, location, code_type);

      converter_.add_symbol(symbol);
    }
  }

  // Handle NumPy functions
  if (is_numpy_call(function_id))
  {
    // Adjust the function ID when reusing functions from the C models
    if (type_utils::is_c_model_func(function_id.get_function()))
    {
      function_id.set_prefix("c:");
      function_id.set_filename("");
    }

    numpy_call_expr numpy_call(function_id, call_, converter_);
    return numpy_call.get();
  }

  function_call_expr call_expr(function_id, call_, converter_);
  return call_expr.get();
}
