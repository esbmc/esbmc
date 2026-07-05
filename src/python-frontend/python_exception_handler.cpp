#include <python-frontend/python_exception_handler.h>
#include <python-frontend/exception_utils.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/python_list.h>
#include <python-frontend/string/string_builder.h>
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
#include <python-frontend/python_expr_builder.h>

using namespace python_expr;

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

  const bool has_finally =
    element.contains("finalbody") && !element["finalbody"].empty();

  // The `else` clause runs only when the try body completes without an
  // exception. ESBMC's try lowering does not model it (it is silently dropped) —
  // a pre-existing limitation independent of this change. Combined with the
  // finally lowering below, which duplicates the finally body on the normal and
  // exception paths, a dropped `else` would also be skipped on the path where
  // finally runs, compounding the unsoundness. So refuse a non-empty `else`
  // only when a finally is present; plain try/except/else keeps its existing
  // behaviour.
  if (has_finally && element.contains("orelse") && !element["orelse"].empty())
    throw std::runtime_error(
      "try/finally with a non-empty else clause is not supported");

  // Python's `finally` runs on every exit path: normal completion, a caught
  // exception, an *uncaught* exception (run finally, then re-raise), and a
  // return/break/continue that leaves the try. We model the first three by
  // duplicating the finally body on the normal path (after the try/catch) and
  // inside a catch-all handler that re-raises. A return/break/continue inside
  // the try body, an except handler, or the finally body would bypass that
  // appended finally and silently change the result, so refuse those with a
  // clean diagnostic rather than return an unsound verdict.
  if (has_finally)
  {
    bool escapes = body_has_escaping_control_flow(element["body"], false) ||
                   body_has_escaping_control_flow(element["finalbody"], false);
    for (const auto &h : element["handlers"])
      if (h.contains("body"))
        escapes = escapes || body_has_escaping_control_flow(h["body"], false);
    if (escapes)
      throw std::runtime_error(
        "try/finally with return/break/continue in the try, except, or finally "
        "body is not supported");
  }

  exprt try_block = converter_.get_block(element["body"]);
  exprt handler = converter_.get_block(element["handlers"]);

  // A bare `except:` already catches every exception, so the fall-through
  // finally below runs after it on the exception path too. Appending a second
  // catch-all here would collide with the user's one in symex's catch_map
  // (both lower to the "ellipsis" exception id, and the later entry wins),
  // dropping the user's handler. So synthesise the finally-rethrow catch-all
  // only when no bare `except:` is present.
  bool has_catch_all = false;
  for (const auto &h : element["handlers"])
    if (h["type"].is_null())
      has_catch_all = true;

  // Build a catch-all handler `{ finally; re-raise; }` so the finally runs on
  // the exception-propagation path. Appended after any specific handlers, it
  // only fires for exceptions none of them caught.
  std::vector<exprt> handler_ops(
    handler.operands().begin(), handler.operands().end());
  if (has_finally && !has_catch_all)
  {
    exprt finally_handler = converter_.get_block(element["finalbody"]);
    finally_handler.type().set("ellipsis", true); // catch-all
    side_effect_exprt rethrow("cpp-throw", empty_typet());
    rethrow.location() = converter_.get_location_from_decl(element);
    codet rethrow_code("expression");
    rethrow_code.operands().push_back(rethrow);
    finally_handler.copy_to_operands(rethrow_code);
    handler_ops.push_back(finally_handler);
  }

  // A valid Python `try` always has at least one handler or a finally, so
  // handler_ops is non-empty and the cpp-catch has the >= 2 operands it needs
  // (the try block plus at least one handler).
  exprt new_expr = codet("cpp-catch");
  new_expr.move_to_operands(try_block);
  for (const auto &op : handler_ops)
    new_expr.copy_to_operands(op);
  block.move_to_operands(new_expr);

  // finally on the normal-completion path (and after a caught exception).
  if (has_finally)
  {
    exprt final_block = converter_.get_block(element["finalbody"]);
    for (const auto &op : final_block.operands())
      block.copy_to_operands(op);
  }
}

// True if `node` contains a return/break/continue that transfers control out of
// the enclosing try. `return` always escapes (we never descend into a nested
// function/lambda, which would capture it); `break`/`continue` escape only when
// not inside a nested loop (`in_loop`). Used to refuse try/finally shapes whose
// appended finally this lowering would skip.
bool python_exception_handler::body_has_escaping_control_flow(
  const nlohmann::json &node,
  bool in_loop)
{
  if (node.is_array())
  {
    for (const auto &stmt : node)
      if (body_has_escaping_control_flow(stmt, in_loop))
        return true;
    return false;
  }
  if (!node.is_object())
    return false;

  const std::string t = node.value("_type", "");
  if (t == "Return")
    return true;
  if (t == "Break" || t == "Continue")
    return !in_loop;
  // Nested functions/lambdas capture every return/break/continue within them.
  if (t == "FunctionDef" || t == "AsyncFunctionDef" || t == "Lambda")
    return false;

  // A loop captures break/continue in its own body/orelse.
  const bool child_in_loop =
    in_loop || t == "For" || t == "AsyncFor" || t == "While";

  for (const char *key : {"body", "orelse", "finalbody"})
    if (
      node.contains(key) &&
      body_has_escaping_control_flow(node[key], child_in_loop))
      return true;
  if (node.contains("handlers") && node["handlers"].is_array())
    for (const auto &h : node["handlers"])
      if (
        h.is_object() && h.contains("body") &&
        body_has_escaping_control_flow(h["body"], child_in_loop))
        return true;
  // NOTE: `match` arms (cases[*].body) are not traversed because `match` is not
  // yet supported by the frontend (a try containing one already errors during
  // conversion). When match support lands, this predicate must recurse into the
  // case bodies, or a return inside a match arm under a try/finally would
  // silently escape the appended finally.
  return false;
}

void python_exception_handler::get_raise_statement(
  const nlohmann::json &element,
  codet &block)
{
  locationt location = converter_.get_location_from_decl(element);

  // Bare 'raise' (Raise.exc is null) re-raises the active exception.
  // Lower it to a cpp-throw with no operand and an empty exception_list;
  // remove_exceptions re-raises from the global exception state (#5075).
  if (element["exc"].is_null())
  {
    side_effect_exprt side("cpp-throw", empty_typet());
    side.location() = location;

    codet code_expr("expression");
    code_expr.operands().push_back(side);
    block.move_to_operands(code_expr);
    return;
  }

  std::string exc_name;

  // Try to extract the exception name from different AST shapes
  if (element["exc"].contains("func") && element["exc"]["func"].contains("id"))
    exc_name = element["exc"]["func"]["id"].get<std::string>();
  else if (element["exc"].contains("id"))
    exc_name = element["exc"]["id"].get<std::string>();
  else if (element["exc"].is_string())
    exc_name = element["exc"].get<std::string>();
  else
    exc_name = ""; // fallback

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
      exc.contains("args") && !exc["args"].empty() && !exc["args"][0].is_null())
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
    raise.copy_to_operands(build_address_of(arg));
  }
  else
  {
    // For custom exceptions:
    // DECL MyException return_value;
    // FUNCTION_CALL:  MyException(&return_value, &"message");
    // Throw MyException return_value;
    raise = converter_.get_expr(element["exc"]);

    // get_function_call() returns the `_init_undefined` sentinel when the
    // raised class (and its bases) define no __init__. Constructors emit
    // this so var-assign can lower `x = MyClass()` to a bare declaration;
    // the raise path has no such shortcut, so synthesize a zero-initialised
    // instance of the class instead. Without this, the sentinel propagates
    // into the cpp-throw operand and migrate aborts with "_init_undefined
    // ... migrate expr failed". Covers user exception hierarchies whose
    // subclasses inherit __init__ from `Exception`.
    if (raise.id() == "_init_undefined")
    {
      if (type.is_empty())
        type = any_type();
      // type_handler_.get_typet returns a symbol_typet referring to the
      // class's tag; gen_zero has no symbol-id branch and would yield a nil
      // expression, which propagates into the cpp-throw operand and makes
      // symex treat the throw as a bare re-throw. Resolve the symbol to the
      // underlying struct before zero-initialising.
      typet resolved = type;
      if (resolved.id() == "symbol")
        resolved = converter_.name_space().follow(type);
      raise = gen_zero(resolved);
      raise.type() = type;
    }
    else if (raise.is_code() && raise.get("statement") == "function_call")
    {
      code_function_callt call =
        to_code_function_call(converter_.convert_expression_to_code(raise));
      // V.3: build the expression-context call in IREP2, back-migrating once.
      expr2tc fn2;
      migrate_expr(call.function(), fn2);
      std::vector<expr2tc> args2;
      args2.reserve(call.arguments().size());
      for (const exprt &a : call.arguments())
      {
        expr2tc a2;
        migrate_expr(a, a2);
        args2.push_back(std::move(a2));
      }
      exprt tmp = migrate_expr_back(
        side_effect_function_call2tc(migrate_type(type), fn2, args2));
      tmp.location() = location;
      raise = tmp;
    }
    else if (
      raise.id() == "sideeffect" && raise.get("statement") == "cpp-throw")
    {
      // A no-argument exception constructor whose class inherits __init__
      // (e.g. `raise E()` where `class E(Exception): pass`) lowers to a bare
      // cpp-throw side-effect rather than a constructed instance. Wrapping it
      // in the cpp-throw built below would nest two throws and hand value_set
      // a non-struct operand, aborting in make_member. Synthesize a
      // zero-initialised instance instead, as the _init_undefined path does
      // for classes with no __init__ at all. (A constructor with arguments or
      // a custom __init__ yields a proper instance and keeps the paths above.)
      if (type.is_empty())
        type = any_type();
      typet resolved = type;
      if (resolved.id() == "symbol")
        resolved = converter_.name_space().follow(type);
      raise = gen_zero(resolved);
      raise.type() = type;
    }
    else
    {
      if (type.is_empty())
        type = any_type();
      if (raise.type() != type)
        raise = build_typecast(raise, type);
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
    // Exception-bind variables (`except E as v:`) are function-local in
    // Python semantics; keeping file_local=true matches the convention
    // used by other Python frontend temp symbols and prevents rw_set's
    // race-eligible-Python-symbol filter from picking them up.
    symbol.file_local = true;
    exception_symbol = converter_.symbol_table().move_symbol_to_context(symbol);
  }

  // Process exception handler body (symbol now exists)
  exprt catch_block = converter_.get_block(element["body"]);

  // Add type and declaration if exception variable was created
  if (exception_symbol != nullptr)
  {
    catch_block.type() = exception_type;
    exprt sym = build_symbol(*exception_symbol);
    code_declt decl(sym);
    exprt decl_code = converter_.convert_expression_to_code(decl);
    decl_code.location() = exception_symbol->location;

    codet::operandst &ops = catch_block.operands();
    ops.insert(ops.begin(), decl_code);
  }
  else
  {
    // Bare 'except:' (no exception type) catches everything.
    // Mark the catch block type as ellipsis so that adjust_catch
    // produces the "ellipsis" exception_id, which remove_exceptions lowers
    // as a catch-all handler.
    catch_block.type().set("ellipsis", true);
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
    code_declt list_decl(build_symbol(list_temp));
    list_decl.location() = location;
    block.move_to_operands(list_decl);

    code_function_callt &func_call =
      static_cast<code_function_callt &>(const_cast<exprt &>(test));
    func_call.lhs() = build_symbol(list_temp);
    block.move_to_operands(func_call);

    list_expr = build_symbol(list_temp);
  }

  // Get list size via __ESBMC_list_size
  const symbolt *size_sym =
    converter_.symbol_table().find_symbol("c:@F@__ESBMC_list_size");
  if (!size_sym)
    throw std::runtime_error("__ESBMC_list_size function not found");

  symbolt &size_result = converter_.create_tmp_symbol(
    element, "$list_size_result$", size_type(), gen_zero(size_type()));
  code_declt size_decl(build_symbol(size_result));
  size_decl.location() = location;
  block.move_to_operands(size_decl);

  code_function_callt size_func_call;
  size_func_call.function() = build_symbol(*size_sym);
  if (list_expr.type().is_pointer())
    size_func_call.arguments().push_back(list_expr);
  else
    size_func_call.arguments().push_back(build_address_of(list_expr));
  size_func_call.lhs() = build_symbol(size_result);
  size_func_call.type() = size_type();
  size_func_call.location() = location;
  block.move_to_operands(size_func_call);

  // Assert size > 0 (size_result and the 0 literal are both size_type).
  exprt assertion =
    build_greater_than(build_symbol(size_result), gen_zero(size_type()));

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
    // V.3: build the always-fail assert condition in IREP2.
    assert_code.assertion() = migrate_expr_back(gen_false_expr());
    assert_code.location() = location;
    assert_code.location().comment("Assertion on None-returning function");
    attach_assert_message(assert_code);
    block.move_to_operands(assert_code);
    return;
  }

  symbolt temp_symbol = create_assert_temp_variable(location);
  converter_.symbol_table().add(temp_symbol);
  exprt temp_var_expr = build_symbol(temp_symbol);

  code_function_callt function_call =
    create_function_call_statement(func_call_expr, temp_var_expr, location);
  block.move_to_operands(function_call);

  exprt assertion_expr;
  if (is_negated)
  {
    // V.3: build `not <result>` in IREP2 (the temp is bool-typed).
    expr2tc tv2;
    migrate_expr(temp_var_expr, tv2);
    assertion_expr = migrate_expr_back(not2tc(tv2));
  }
  else
  {
    // V.3: build `(int)<temp> == 1` in IREP2 via the build_equal helper (the
    // temp is a synthetic bool; both operands are signedbv 32), mirroring the
    // is_negated `not` branch above.
    exprt cast_expr = build_typecast(temp_var_expr, signedbv_typet(32));
    exprt one_expr = constant_exprt("1", signedbv_typet(32));
    assertion_expr = build_equal(cast_expr, one_expr);
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
  temp_symbol.set_type(bool_type());
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
