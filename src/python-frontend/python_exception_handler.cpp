#include <python-frontend/python_exception_handler.h>
#include <python-frontend/exception_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_list.h>
#include <python-frontend/string_builder.h>
#include <python-frontend/symbol_id.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/type_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/expr_util.h>
#include <util/message.h>
#include <util/python_types.h>
#include <util/std_code.h>
#include <util/string_constant.h>

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

python_exception_handler::python_exception_handler(
  python_converter &converter,
  type_handler &type_handler)
  : converter_(converter), type_handler_(type_handler)
{
}

// ---------------------------------------------------------------------------
// Statement converters
// ---------------------------------------------------------------------------

void python_exception_handler::get_try_statement(
  const nlohmann::json &element,
  codet &block)
{
  // Check if this try block wraps a failed import (module_not_found = true)
  // and has an ImportError handler.  If so, statically take the except branch.
  bool has_missing_import = false;
  for (const auto &stmt : element["body"])
  {
    if (
      (stmt["_type"] == "Import" || stmt["_type"] == "ImportFrom") &&
      stmt.value("module_not_found", false))
    {
      has_missing_import = true;
      break;
    }
  }

  if (has_missing_import)
  {
    for (const auto &handler : element["handlers"])
    {
      if (
        !handler["type"].is_null() &&
        handler["type"].value("id", "") == "ImportError")
      {
        // Directly emit only the except-handler body
        exprt except_body = converter_.get_block(handler["body"]);
        for (const auto &op : except_body.operands())
          block.copy_to_operands(op);
        break;
      }
    }
    return;
  }

  exprt new_expr = codet("cpp-catch");
  exprt try_block = converter_.get_block(element["body"]);
  exprt handler = converter_.get_block(element["handlers"]);
  new_expr.move_to_operands(try_block);

  for (const auto &op : handler.operands())
    new_expr.copy_to_operands(op);

  block.move_to_operands(new_expr);
}

void python_exception_handler::get_raise_statement(
  const nlohmann::json &element,
  codet &block)
{
  std::string exc_name;

  // Try to extract the exception name from different AST shapes
  if (
    element["exc"].contains("func") &&
    element["exc"]["func"].contains("id"))
    exc_name = element["exc"]["func"]["id"].get<std::string>();
  else if (element["exc"].contains("id"))
    exc_name = element["exc"]["id"].get<std::string>();
  else if (element["exc"].is_string())
    exc_name = element["exc"].get<std::string>();
  else
    exc_name = ""; // fallback

  locationt location = converter_.get_location_from_decl(element);
  typet type = type_handler_.get_typet(exc_name);

  // AssertionError is special-cased to a clean assert(false)
  if (exc_name == "AssertionError")
  {
    code_assertt assert_code{false_exprt()};
    assert_code.location() = location;
    if (
      element["exc"].contains("args") && !element["exc"]["args"].empty() &&
      !element["exc"]["args"][0].is_null())
    {
      const std::string msg =
        converter_.get_string_handler().process_format_spec(
          element["exc"]["args"][0]);
      assert_code.location().comment(msg);
    }
    block.move_to_operands(assert_code);
    return;
  }

  exprt raise;
  if (type_utils::is_python_exceptions(exc_name))
  {
    // Construct a constant struct to throw: raise { .message=&"Error message" }
    exprt arg;
    const auto &exc = element["exc"];
    if (
      exc.contains("args") && !exc["args"].empty() &&
      !exc["args"][0].is_null())
    {
      const auto &json_arg = exc["args"][0];
      exprt tmp = converter_.get_expr(json_arg);
      arg = string_constantt(
        converter_.get_string_handler().process_format_spec(json_arg),
        tmp.type(),
        string_constantt::k_default);
    }
    else
    {
      // No arguments: create default empty message
      arg = string_constantt(
        "",
        type_handler_.build_array(char_type(), 1),
        string_constantt::k_default);
    }

    raise.id("struct");
    raise.type() = type;
    raise.copy_to_operands(address_of_exprt(arg));
  }
  else
  {
    // For custom exceptions:
    // DECL MyException return_value;
    // FUNCTION_CALL:  MyException(&return_value, &"message");
    // Throw MyException return_value;
    raise = converter_.get_expr(element["exc"]);
    if (raise.is_code() && raise.get("statement") == "function_call")
    {
      code_function_callt call =
        to_code_function_call(converter_.convert_expression_to_code(raise));
      side_effect_expr_function_callt tmp;
      tmp.function() = call.function();
      tmp.arguments() = call.arguments();
      tmp.type() = type;
      tmp.location() = location;
      raise = tmp;
    }
    else
    {
      if (type.is_empty())
        type = any_type();
      if (raise.type() != type)
        raise = typecast_exprt(raise, type);
    }
  }

  side_effect_exprt side("cpp-throw", type);
  side.location() = location;
  side.move_to_operands(raise);

  codet code_expr("expression");
  code_expr.operands().push_back(side);
  block.move_to_operands(code_expr);
}

void python_exception_handler::get_except_handler_statement(
  const nlohmann::json &element,
  codet &block)
{
  symbolt *exception_symbol = nullptr;
  typet exception_type;

  // Create exception variable symbol before processing body
  if (!element["type"].is_null())
  {
    exception_type =
      type_handler_.get_typet(element["type"]["id"].get<std::string>());

    std::string name;
    symbol_id sid = converter_.create_symbol_id();
    locationt location = converter_.get_location_from_decl(element);
    std::string module_name = location.get_file().as_string();

    // Check if the exception handler binds the exception to a variable
    if (!element["name"].is_null())
      name = element["name"].get<std::string>();
    else
      name = "__anon_exc_var_" + location.get_line().as_string();

    sid.set_object(name);

    symbolt symbol = converter_.create_symbol(
      module_name,
      converter_.current_function_name(),
      sid.to_string(),
      location,
      exception_type);
    symbol.name = name;
    symbol.lvalue = true;
    symbol.is_extern = false;
    symbol.file_local = false;
    exception_symbol = converter_.symbol_table().move_symbol_to_context(symbol);
  }

  // Process exception handler body (symbol now exists)
  exprt catch_block = converter_.get_block(element["body"]);

  // Add type and declaration if exception variable was created
  if (exception_symbol != nullptr)
  {
    catch_block.type() = exception_type;
    exprt sym = symbol_expr(*exception_symbol);
    code_declt decl(sym);
    exprt decl_code = converter_.convert_expression_to_code(decl);
    decl_code.location() = exception_symbol->location;

    codet::operandst &ops = catch_block.operands();
    ops.insert(ops.begin(), decl_code);
  }

  block.move_to_operands(catch_block);
}

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------

void python_exception_handler::handle_list_assertion(
  const nlohmann::json &element,
  const exprt &test,
  code_blockt &block,
  const std::function<void(code_assertt &)> &attach_assert_message)
{
  locationt location = converter_.get_location_from_decl(element);

  // Materialise function call if needed
  exprt list_expr = test;
  if (test.is_function_call())
  {
    symbolt &list_temp = converter_.create_tmp_symbol(
      element, "$list_assert_temp$", test.type(), exprt());
    code_declt list_decl(symbol_expr(list_temp));
    list_decl.location() = location;
    block.move_to_operands(list_decl);

    code_function_callt &func_call =
      static_cast<code_function_callt &>(const_cast<exprt &>(test));
    func_call.lhs() = symbol_expr(list_temp);
    block.move_to_operands(func_call);

    list_expr = symbol_expr(list_temp);
  }

  // Get list size via __ESBMC_list_size
  const symbolt *size_sym =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
  if (!size_sym)
    throw std::runtime_error("__ESBMC_list_size function not found");

  symbolt &size_result = converter_.create_tmp_symbol(
    element, "$list_size_result$", size_type(), gen_zero(size_type()));
  code_declt size_decl(symbol_expr(size_result));
  size_decl.location() = location;
  block.move_to_operands(size_decl);

  code_function_callt size_func_call;
  size_func_call.function() = symbol_expr(*size_sym);
  if (list_expr.type().is_pointer())
    size_func_call.arguments().push_back(list_expr);
  else
    size_func_call.arguments().push_back(address_of_exprt(list_expr));
  size_func_call.lhs() = symbol_expr(size_result);
  size_func_call.type() = size_type();
  size_func_call.location() = location;
  block.move_to_operands(size_func_call);

  // Assert size > 0
  exprt assertion(">", bool_type());
  assertion.copy_to_operands(symbol_expr(size_result), gen_zero(size_type()));

  code_assertt assert_code;
  assert_code.assertion() = assertion;
  assert_code.location() = location;
  attach_assert_message(assert_code);
  block.move_to_operands(assert_code);
}

void python_exception_handler::handle_function_call_assertion(
  const nlohmann::json &element,
  const exprt &func_call_expr,
  bool is_negated,
  code_blockt &block,
  const std::function<void(code_assertt &)> &attach_assert_message)
{
  locationt location = converter_.get_location_from_decl(element);
  const typet &return_type = func_call_expr.type();

  if (return_type == none_type() || return_type.id() == "empty")
  {
    // Function returns None: execute call and assert False
    exprt func_call_copy = func_call_expr;
    codet code_stmt = converter_.convert_expression_to_code(func_call_copy);
    block.move_to_operands(code_stmt);

    code_assertt assert_code;
    assert_code.assertion() = false_exprt();
    assert_code.location() = location;
    assert_code.location().comment("Assertion on None-returning function");
    attach_assert_message(assert_code);
    block.move_to_operands(assert_code);
    return;
  }

  symbolt temp_symbol = create_assert_temp_variable(location);
  converter_.symbol_table().add(temp_symbol);
  exprt temp_var_expr = symbol_expr(temp_symbol);

  code_function_callt function_call =
    create_function_call_statement(func_call_expr, temp_var_expr, location);
  block.move_to_operands(function_call);

  exprt assertion_expr;
  if (is_negated)
  {
    assertion_expr = not_exprt(temp_var_expr);
  }
  else
  {
    exprt cast_expr = typecast_exprt(temp_var_expr, signedbv_typet(32));
    exprt one_expr = constant_exprt("1", signedbv_typet(32));
    assertion_expr = equality_exprt(cast_expr, one_expr);
  }

  code_assertt assert_code;
  assert_code.assertion() = assertion_expr;
  assert_code.location() = location;
  attach_assert_message(assert_code);
  block.move_to_operands(assert_code);
}

// ---------------------------------------------------------------------------
// Expression helper
// ---------------------------------------------------------------------------

exprt python_exception_handler::gen_exception_raise(
  const std::string &exc,
  const std::string &message) const
{
  return python_exception_utils::make_exception_raise(
    type_handler_, exc, message, nullptr);
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

symbolt python_exception_handler::create_assert_temp_variable(
  const locationt &location) const
{
  symbol_id temp_sid = converter_.create_symbol_id();
  temp_sid.set_object("__assert_temp");
  const std::string temp_sid_str = temp_sid.to_string();

  symbolt temp_symbol;
  temp_symbol.id = temp_sid_str;
  temp_symbol.name = temp_sid_str;
  temp_symbol.type = bool_type();
  temp_symbol.lvalue = true;
  temp_symbol.static_lifetime = false;
  temp_symbol.location = location;
  return temp_symbol;
}

code_function_callt python_exception_handler::create_function_call_statement(
  const exprt &func_call_expr,
  const exprt &lhs_var,
  const locationt &location)
{
  code_function_callt function_call;
  function_call.lhs() = lhs_var;
  function_call.function() = func_call_expr.operands()[1];

  const exprt &args_operand = func_call_expr.operands()[2];
  code_function_callt::argumentst arguments;
  for (const auto &arg : args_operand.operands())
    arguments.push_back(arg);

  function_call.arguments() = arguments;
  function_call.location() = location;
  return function_call;
}
