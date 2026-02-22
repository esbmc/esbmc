#pragma once

#include <util/std_code.h>
#include <util/expr.h>
#include <util/message.h>

#include <nlohmann/json.hpp>
#include <functional>
#include <string>

class python_converter;
class type_handler;

/**
 * @brief Handles Python exception-related statement conversion.
 *
 * Centralises all try/except/raise statement conversion and exception
 * expression helpers that were previously scattered across python_converter
 * and function_call_expr.  Keeping exception logic here makes python_converter
 * easier to read and makes it straightforward to extend exception support in
 * one place.
 */
class python_exception_handler
{
public:
  explicit python_exception_handler(
    python_converter &converter,
    type_handler &type_handler);

  // -----------------------------------------------------------------------
  // Statement converters (called from python_converter::get_block)
  // -----------------------------------------------------------------------

  /**
   * Convert a Python try/except block.
   *
   * Handles the special case where the try body contains a failed import
   * (module_not_found flag set by the pre-processor) and statically selects
   * only the ImportError handler body.  For all other cases the full
   * cpp-catch IR node is produced.
   *
   * @param element  The AST node with _type == "Try".
   * @param block    The target code block to receive the converted statement.
   */
  void get_try_statement(const nlohmann::json &element, codet &block);

  /**
   * Convert a Python raise statement.
   *
   * Raises are lowered to side-effect cpp-throw expressions.  AssertionError
   * is special-cased to a code_assertt(false) for cleaner verification output.
   *
   * @param element  The AST node with _type == "Raise".
   * @param block    The target code block to receive the converted statement.
   */
  void get_raise_statement(const nlohmann::json &element, codet &block);

  /**
   * Convert a Python except-handler clause.
   *
   * Creates a symbol for the bound exception variable (if any), processes the
   * handler body and prepends the declaration.
   *
   * @param element  The AST node with _type == "ExceptHandler".
   * @param block    The target code block to receive the converted statement.
   */
  void get_except_handler_statement(
    const nlohmann::json &element,
    codet &block);

  // -----------------------------------------------------------------------
  // Assertion helpers (previously python_converter private methods)
  // -----------------------------------------------------------------------

  /**
   * Emit a truthiness assertion on a list value.
   *
   * Materialises the list (handling function-call RHS), calls
   * __ESBMC_list_size and asserts size > 0.
   *
   * @param element              AST node used for location information.
   * @param test                 Expression representing the list.
   * @param block                Target block.
   * @param attach_assert_message Callback that sets the comment on the
   *                             produced code_assertt.
   */
  void handle_list_assertion(
    const nlohmann::json &element,
    const exprt &test,
    code_blockt &block,
    const std::function<void(code_assertt &)> &attach_assert_message);

  /**
   * Emit an assertion whose condition depends on a function-call result.
   *
   * Creates a temporary boolean variable, executes the call and asserts the
   * result (or its negation when @p is_negated is true).  For functions that
   * return None the call is executed and then False is asserted.
   *
   * @param element              AST node used for location information.
   * @param func_call_expr       The function-call expression.
   * @param is_negated           True when the test was wrapped in a UnaryOp Not.
   * @param block                Target block.
   * @param attach_assert_message Callback to set the assertion comment.
   */
  void handle_function_call_assertion(
    const nlohmann::json &element,
    const exprt &func_call_expr,
    bool is_negated,
    code_blockt &block,
    const std::function<void(code_assertt &)> &attach_assert_message);

  // -----------------------------------------------------------------------
  // Expression helpers (previously function_call_expr / exception_utils)
  // -----------------------------------------------------------------------

  /**
   * Build a cpp-throw side-effect expression for a named Python exception.
   *
   * @param exc     Exception class name, e.g. "TypeError".
   * @param message Human-readable message string.
   * @return        A side_effect_exprt of type cpp-throw.
   */
  exprt gen_exception_raise(
    const std::string &exc,
    const std::string &message) const;

private:
  python_converter &converter_;
  type_handler &type_handler_;

  // ------------------------------------------------------------------
  // Internal helpers
  // ------------------------------------------------------------------

  /** Create a temporary boolean symbol used by assertion helpers. */
  symbolt create_assert_temp_variable(const locationt &location) const;

  /**
   * Build a code_function_callt for a function-call expression, setting the
   * provided @p lhs_var as the return destination.
   */
  static code_function_callt create_function_call_statement(
    const exprt &func_call_expr,
    const exprt &lhs_var,
    const locationt &location);
};
