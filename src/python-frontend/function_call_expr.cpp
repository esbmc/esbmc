#include <python-frontend/function_call_expr.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/json_utils.h>
#include <util/c_typecast.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/string_constant.h>
#include <regex>

using namespace json_utils;

const std::string kGetObjectSize = "__ESBMC_get_object_size";
const std::string kEsbmcAssume = "__ESBMC_assume";
const std::string kVerifierAssume = "__VERIFIER_assume";

function_call_expr::function_call_expr(
  const nlohmann::json &call,
  python_converter &converter)
  : call_(call),
    converter_(converter),
    type_handler_(converter.get_type_handler()),
    function_type_(FunctionType::FreeFunction)
{
  build_function_id();

  get_function_type();

  // Add assume and len functions to symbol table
  if (is_assume_call() || is_len_call())
  {
    const auto &symbol_table = converter_.symbol_table();
    const std::string &func_symbol_id = function_id_.to_string();

    if (symbol_table.find_symbol(func_symbol_id.c_str()) == nullptr)
    {
      code_typet code_type;
      if (is_len_call())
      {
        code_type.return_type() = int_type();
        code_type.arguments().push_back(pointer_typet(empty_typet()));
      }

      const std::string &python_file = converter_.python_file();
      const std::string &func_name = function_id_.get_function();
      locationt location = converter_.get_location_from_decl(call_);

      symbolt symbol = converter_.create_symbol(
        python_file, func_name, func_symbol_id, location, code_type);

      converter_.add_symbol(symbol);
    }
  }
}

static std::string get_classname_from_symbol_id(const std::string &symbol_id)
{
  // This function might return "Base" for a symbol_id as: py:main.py@C@Base@F@foo@self

  std::string class_name;
  size_t class_pos = symbol_id.find("@C@");
  size_t func_pos = symbol_id.find("@F@");

  if (class_pos != std::string::npos && func_pos != std::string::npos)
  {
    size_t length = func_pos - (class_pos + 3); // "+3" to ignore "@C@"
    // Extract substring between "@C@" and "@F@"
    class_name = symbol_id.substr(class_pos + 3, length);
  }
  return class_name;
}

void function_call_expr::get_function_type()
{
  if (type_handler_.is_constructor_call(call_))
  {
    function_type_ = FunctionType::Constructor;
    return;
  }

  if (call_["func"]["_type"] == "Attribute")
  {
    std::string caller = get_object_name();

    // Handling a function call as a class method call when:
    // (1) The caller corresponds to a class name, for example: MyClass.foo().
    // (2) Calling methods of built-in types, such as int.from_bytes()
    //     All the calls to built-in methods are handled by class methods in operational models.
    // (3) Calling a instance method from a built-in type object, for example: x.bit_length() when x is an int
    // If the caller is a class or a built-in type, the following condition detects a class method call.
    if (
      is_class(caller, converter_.ast()) ||
      type_utils::is_builtin_type(caller) ||
      type_utils::is_builtin_type(type_handler_.get_var_type(caller)))
    {
      function_type_ = FunctionType::ClassMethod;
    }
    else if (!converter_.is_imported_module(caller))
    {
      function_type_ = FunctionType::InstanceMethod;
    }
  }
}

void function_call_expr::build_function_id()
{
  const std::string &python_file = converter_.python_file();
  const std::string &current_class_name = converter_.current_classname();
  const std::string &current_function_name = converter_.current_function_name();
  const auto &ast = converter_.ast();

  bool is_member_function_call = false;

  const auto &func_json = call_["func"];

  const std::string &func_type = func_json["_type"];

  std::string func_name, obj_name, class_name;

  function_id_ =
    symbol_id(python_file, current_class_name, current_function_name);

  if (func_type == "Name")
  {
    func_name = func_json["id"];
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
      std::string lhs_type =
        type_handler_.get_operand_type(func_json["value"]["left"]);

      std::string rhs_type =
        type_handler_.get_operand_type(func_json["value"]["right"]);

      assert(lhs_type == rhs_type);

      obj_name = lhs_type;
    }
    else
    {
      obj_name = func_json["value"]["id"];
    }

    obj_name = get_object_alias(ast, obj_name);

    if (!is_class(obj_name, ast) && converter_.is_imported_module(obj_name))
    {
      const auto &module_path = converter_.get_imported_module_path(obj_name);

      function_id_ =
        symbol_id(module_path, current_class_name, current_function_name);

      is_member_function_call = false;
    }
  }

  // build symbol_id
  if (func_name == "len")
  {
    func_name = kGetObjectSize;
    function_id_.clear();
    function_id_.set_prefix("c:");
  }
  else if (type_utils::is_builtin_type(obj_name))
  {
    class_name = obj_name;
    function_id_ = symbol_id(python_file, class_name, func_name);
  }
  else if (is_assume_call())
  {
    function_id_.clear();
  }

  // Insert class name in the symbol id
  if (type_handler_.is_constructor_call(call_))
  {
    class_name = func_name;
  }
  else if (is_member_function_call)
  {
    if (type_utils::is_builtin_type(obj_name) || is_class(obj_name, ast))
    {
      class_name = obj_name;
    }
    else
    {
      auto obj_node = find_var_decl(obj_name, current_function_name, ast);

      if (obj_node.empty())
        throw std::runtime_error("Class " + obj_name + " not found");

      class_name = obj_node["annotation"]["id"].get<std::string>();
    }
  }

  if (!class_name.empty())
  {
    function_id_.set_class(class_name);
  }

  function_id_.set_function(func_name);
}

bool function_call_expr::is_nondet_call() const
{
  std::regex pattern(
    R"(nondet_(int|char|bool|float)|__VERIFIER_nondet_(int|char|bool|float))");

  return std::regex_match(function_id_.get_function(), pattern);
}

bool function_call_expr::is_assume_call() const
{
  const std::string &func_name = function_id_.get_function();
  return (func_name == kEsbmcAssume || func_name == kVerifierAssume);
}

bool function_call_expr::is_len_call() const
{
  const std::string &func_name = function_id_.get_function();
  return func_name == kGetObjectSize;
}

exprt function_call_expr::build_nondet_call() const
{
  const std::string &func_name = function_id_.get_function();

  // Function name pattern: nondet_(type). e.g: nondet_bool(), nondet_int()
  size_t underscore_pos = func_name.rfind("_");
  std::string type = func_name.substr(underscore_pos + 1);
  exprt rhs = exprt("sideeffect", type_handler_.get_typet(type));
  rhs.statement("nondet");
  return rhs;
}

exprt function_call_expr::build_constant_from_arg() const
{
  const std::string &func_name = function_id_.get_function();

  size_t arg_size = 1;
  auto arg = call_["args"][0];

  if (func_name == "str")
    arg_size = arg["value"].get<std::string>().size(); // get string length

  else if (func_name == "int" && arg["value"].is_number_float())
  {
    double arg_value = arg["value"].get<double>();
    arg["value"] = static_cast<int>(arg_value);
  }

  typet t = type_handler_.get_typet(func_name, arg_size);
  exprt expr = converter_.get_expr(arg);
  expr.type() = t;
  return expr;
}

std::string function_call_expr::get_object_name() const
{
  const auto &subelement = call_["func"]["value"];

  std::string obj_name;
  if (subelement["_type"] == "Attribute")
    obj_name = subelement["attr"].get<std::string>();
  else if (subelement["_type"] == "Constant" || subelement["_type"] == "BinOp")
    obj_name = function_id_.get_class();
  else
    obj_name = subelement["id"].get<std::string>();

  return json_utils::get_object_alias(converter_.ast(), obj_name);
}

bool function_call_expr::is_numpy_call() const
{
  const std::string &filename = function_id_.get_filename();
  const std::string &suffix = "/models/numpy.py";

  return (filename.rfind(suffix) == (filename.size() - suffix.size()));
}

exprt function_call_expr::build()
{
  if (is_numpy_call()) {
	  printf("is numpy call\n");
  }

  // Handle non-det functions
  if (is_nondet_call())
  {
    return build_nondet_call();
  }

  const std::string &func_name = function_id_.get_function();

  /* Calls to initialise variables using built-in type functions such as int(1), str("test"), bool(1)
   * are converted to simple variable assignments, simplifying the handling of built-in type objects.
   * For example, x = int(1) becomes x = 1. */
  if (
    type_utils::is_builtin_type(func_name) ||
    type_utils::is_consensus_type(func_name))
  {
    return build_constant_from_arg();
  }

  auto &symbol_table = converter_.symbol_table();

  // Get object symbol
  symbolt *obj_symbol = nullptr;
  symbol_id obj_symbol_id = converter_.create_symbol_id();

  if (call_["func"]["_type"] == "Attribute")
  {
    std::string caller = get_object_name();
    obj_symbol_id.set_object(caller);
    obj_symbol = symbol_table.find_symbol(obj_symbol_id.to_string());
  }

  // Get function symbol
  const std::string &func_symbol_id = function_id_.to_string();
  assert(!func_symbol_id.empty());

  // Find function in the current module
  const symbolt *func_symbol = symbol_table.find_symbol(func_symbol_id.c_str());

  // Find function in imported modules
  if (!func_symbol)
    func_symbol = converter_.find_imported_symbol(func_symbol_id);

  if (func_symbol == nullptr)
  {
    if (
      function_type_ == FunctionType::Constructor ||
      function_type_ == FunctionType::InstanceMethod)
    {
      // Get method from a base class when it is not defined in the current class
      func_symbol = converter_.find_function_in_base_classes(
        function_id_.get_class(),
        func_symbol_id,
        func_name,
        function_type_ == FunctionType::Constructor);

      if (function_type_ == FunctionType::Constructor)
      {
        if (!func_symbol)
        {
          // If __init__() is not defined for the class and bases,
          // an assignment (x = MyClass()) is converted to a declaration (x:MyClass) in python_converter::get_var_assign().
          return exprt("_init_undefined");
        }
        converter_.base_ctor_called = true;
      }
      else if (function_type_ == FunctionType::InstanceMethod)
      {
        assert(obj_symbol);

        // Update obj attributes from self
        converter_.update_instance_from_self(
          get_classname_from_symbol_id(func_symbol->id.as_string()),
          func_name,
          obj_symbol_id.to_string());
      }
    }
    else
    {
      log_warning("Undefined function: {}", func_name.c_str());
      return exprt();
    }
  }

  locationt location = converter_.get_location_from_decl(call_);

  code_function_callt call;
  call.location() = location;
  call.function() = symbol_expr(*func_symbol);
  const typet &return_type = to_code_type(func_symbol->type).return_type();
  call.type() = return_type;

  // Add self as first parameter
  if (function_type_ == FunctionType::Constructor)
  {
    // Self is the LHS
    assert(converter_.ref_instance);
    call.arguments().push_back(gen_address_of(*converter_.ref_instance));
  }
  else if (function_type_ == FunctionType::InstanceMethod)
  {
    assert(obj_symbol);
    // Passing object as "self" (first) parameter on instance method calls
    call.arguments().push_back(gen_address_of(symbol_expr(*obj_symbol)));
  }
  else if (function_type_ == FunctionType::ClassMethod)
  {
    // Passing a void pointer to the "cls" argument
    typet t = pointer_typet(empty_typet());
    call.arguments().push_back(gen_zero(t));

    // All methods for the int class without parameters acts solely on the encapsulated integer value.
    // Therefore, we always pass the caller (obj) as a parameter in these functions.
    // For example, if x is an int instance, x.bit_length() call becomes bit_length(x)
    if (
      obj_symbol &&
      type_handler_.get_var_type(obj_symbol->name.as_string()) == "int" &&
      call_["args"].empty())
    {
      call.arguments().push_back(symbol_expr(*obj_symbol));
    }
    else if (call_["func"]["value"]["_type"] == "BinOp")
    {
      // Handling function call from binary expressions like: (x+1).bit_length()
      call.arguments().push_back(converter_.get_expr(call_["func"]["value"]));
    }
  }

  for (const auto &arg_node : call_["args"])
  {
    exprt arg = converter_.get_expr(arg_node);
    if (is_len_call())
    {
      c_typecastt c_typecast(converter_.name_space());
      c_typecast.implicit_typecast(arg, pointer_typet(empty_typet()));
    }

    // All array function arguments (e.g. bytes type) are handled as pointers.
    if (arg.type().is_array())
    {
      if (arg_node["_type"] == "Constant" && arg_node["value"].is_string())
      {
        arg = string_constantt(
          arg_node["value"].get<std::string>(),
          arg.type(),
          string_constantt::k_default);
      }
      call.arguments().push_back(address_of_exprt(arg));
    }
    else
      call.arguments().push_back(arg);
  }

  if (is_len_call())
  {
    side_effect_expr_function_callt sideeffect;
    sideeffect.function() = call.function();
    sideeffect.arguments() = call.arguments();
    sideeffect.location() = call.location();
    sideeffect.type() =
      static_cast<const typet &>(call.function().type().return_type());
    return sideeffect;
  }

  return call;
}
