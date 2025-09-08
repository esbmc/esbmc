#pragma once

#include <python-frontend/python_converter.h>
#include <python-frontend/type_handler.h>
#include <python-frontend/symbol_id.h>
#include <util/expr.h>
#include <nlohmann/json.hpp>

enum class FunctionType
{
  Constructor,
  ClassMethod,
  InstanceMethod,
  FreeFunction,
};

class symbol_id;

class function_call_expr
{
public:
  function_call_expr(
    const symbol_id &function_id,
    const nlohmann::json &call,
    python_converter &converter);

  virtual ~function_call_expr() = default;

  /*
   * Converts the function from the AST into an exprt.
   */
  virtual exprt get();

  const symbol_id &get_function_id() const
  {
    return function_id_;
  }

private:
  /**
   * Determines whether a non-deterministic function is being invoked.
   */
  bool is_nondet_call() const;

  bool is_introspection_call() const;

  bool is_input_call() const;

  // Create an expression that represents non-deterministic string input
  exprt handle_input() const;

  // Helper method for UTF-8 logic
  int decode_utf8_codepoint(const std::string &utf8_str) const;

  /*
   * Creates an expression for a non-deterministic function call.
   */
  exprt build_nondet_call() const;

  /*
   * Creates a constant expression from function argument.
   */
  exprt build_constant_from_arg() const;

  /*
   * Sets the function_type_ attribute based on the call information.
   */
  void get_function_type();

  /*
   * Retrieves the object (caller) name from the AST.
   */
  std::string get_object_name() const;

  /*
   * Handles int-to-str conversions (e.g., str(65)) by generating
   * the appropriate cast expression.
   */
  exprt handle_int_to_str(nlohmann::json &arg) const;

  /*
   * Extracts a string representation from a symbol's constant value.
   * Handles both character arrays (e.g., ['6', '5']) and single-character
   * constants by decoding their binary-encoded bitvector representations.
   */
  std::optional<std::string>
  extract_string_from_symbol(const symbolt *sym) const;

  /*
   * Looks up a Python variable's symbol using its identifier and the
   * current filename to construct the full scoped symbol name.
   */
  const symbolt *lookup_python_symbol(const std::string &var_name) const;

  exprt handle_isinstance() const;

  exprt handle_hasattr() const;

  /*
   * Handles str-to-int conversions (e.g., int('65')) by reconstructing
   * the string value from a symbol's internal representation and
   * converting it to an integer expression.
   */
  exprt handle_str_symbol_to_int(const symbolt *sym) const;

  /*
   * Handles str-to-float conversions (e.g., float("3.14")) by reconstructing
   * the string value from the symbol and converting it to a float expression.
   */
  exprt handle_str_symbol_to_float(const symbolt *sym) const;

  /*
   * Handles float-to-str conversions (e.g., str(5.5)) by converting
   * the float value into a string representation and generating
   * the corresponding constant character array expression.
   */
  exprt handle_float_to_str(nlohmann::json &arg) const;

  /*
   * Handles string arguments (e.g., str("abc")) by converting them
   * into character array expressions.
   */
  size_t handle_str(nlohmann::json &arg) const;

  /*
   * Handles float-to-int conversions (e.g., int(3.14)) by generating
   * the appropriate cast expression.
   */
  void handle_float_to_int(nlohmann::json &arg) const;

  /*
   * Handles int-to-float conversions (e.g., float(3)) by generating
   * the appropriate cast expression.
   */
  void handle_int_to_float(nlohmann::json &arg) const;

  /*
   * Handles chr(int) conversions by creating a single-character
   * string expression from an integer.
   */
  exprt handle_chr(nlohmann::json &arg) const;

  /*
   * Handles hexadecimal string arguments (e.g., hex(255) -> "0xff")
   * by building a constant expression representing the string.
   */
  exprt handle_hex(nlohmann::json &arg) const;

  /*
   * Handles octal string arguments (e.g., oct(8) -> "0o10")
   * by building a constant expression representing the resulting
   * string. Supports both positive and negative integers,
   * following the Python 3 built-in `oct()` function semantics.
   */
  exprt handle_oct(nlohmann::json &arg) const;

  /*
   * Handles ord(str) conversions by extracting the Unicode code point
   * (as an integer) from a single-character string expression.
   */
  exprt handle_ord(nlohmann::json &arg) const;

  /*
   * Handles abs() function calls by computing the absolute value of the argument.
   * The argument can be an integer, a floating-point number, or an object implementing
   * the __abs__() method. The function returns an expression representing the absolute value.
   */
  exprt handle_abs(nlohmann::json &arg) const;

  /*
   * Checks if the current function call is a min() or max() built-in function.
   * Returns true if the function name matches "min" or "max", false otherwise.
   */
  bool is_min_max_call() const;

  /*
   * Handles min() or max() function calls by generating conditional expressions.
   * Currently supports exactly 2 arguments.
   * @TODO: Support multiple arguments.
   * For min(a, b), generates: a < b ? a : b
   * For max(a, b), generates: a > b ? a : b
   * Performs type compatibility checking with automatic int-to-float promotion.
   */
  exprt
  handle_min_max(const std::string &func_name, irep_idt comparison_op) const;

  /*
   * Handles round() function calls by rounding numeric values to the nearest integer.
   * Only supports single-argument form: round(x)
   * - round(x): returns nearest integer as int type
   * - round(x, ndigits): NOT SUPPORTED - throws error
   * Uses "round half to even" (banker's rounding) for tie-breaking.
   */
  exprt handle_round(const nlohmann::json &args) const;

  /*
   * Helper function to create symbolic round expression using C round() function.
   * Converts the operand to double, calls C round(), then casts result back to int.
   */
  exprt handle_round_symbolic(exprt operand) const;

protected:
  symbol_id function_id_;
  const nlohmann::json &call_;
  python_converter &converter_;
  const type_handler &type_handler_;
  FunctionType function_type_;
};
