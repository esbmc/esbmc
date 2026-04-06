#ifndef PYTHON_FRONTEND_COMPLEX_HANDLER_H
#define PYTHON_FRONTEND_COMPLEX_HANDLER_H

#include <nlohmann/json.hpp>
#include <util/expr.h>
#include <string>
#include <unordered_map>

class python_converter;
class contextt;
class type_handler;
class symbolt;

/**
 * @brief Dedicated handler for complex number operations in the Python frontend.
 *
 * Centralises complex arithmetic (mul, div, pow, log, exp), unary operations,
 * attribute access (.real, .imag, .conjugate()), abs(), and numeric
 * normalisation that were previously scattered across python_converter.cpp
 * and function_call_expr.cpp.
 *
 * Follows the pattern established by string_handler and python_math.
 */
class complex_handler
{
public:
  complex_handler(
    python_converter &converter,
    contextt &symbol_table,
    type_handler &type_handler_ref);

  /**
   * Handles binary operations when at least one operand is complex.
   * Covers Add, Sub, Mult, Div, Pow, Eq, NotEq and rejects
   * unsupported comparisons/floor ops with proper TypeErrors.
   */
  exprt handle_binary_op(
    const std::string &op,
    const exprt &lhs,
    const exprt &rhs,
    const nlohmann::json &element) const;

  /**
   * Handles unary +/- on complex values.
   * UAdd returns the operand unchanged; USub negates both components.
   */
  exprt handle_unary_op(const std::string &op, const exprt &operand) const;

  /**
   * Handles attribute access / method calls on complex objects.
   * Dispatches conjugate() when the method name matches.
   *
   * @param element The full JSON node of the Call expression
   *                (must contain func.attr and func.value).
   * @return The resulting expression, or nil_exprt if the call is not
   *         a recognised complex method.
   */
  exprt handle_attribute(const nlohmann::json &element) const;

  /**
   * Computes abs(z) = sqrt(real^2 + imag^2).
   */
  exprt handle_abs(const exprt &z) const;

  /**
   * Handles cmath.log / cmath.log10 with complex arguments.
   *
   * @param func_name "log" or "log10".
   * @param call      The full Call JSON node (used for location info).
   * @param args      Positional arguments array.
   * @param keywords  Keyword arguments array.
   * @return The resulting complex expression, or a TypeError throw.
   */
  exprt handle_cmath_log(
    const std::string &func_name,
    const nlohmann::json &call,
    const nlohmann::json &args,
    const nlohmann::json &keywords) const;

  /**
   * Normalises a numeric expression for use in complex arithmetic.
   * Integer/bool trees are recursively promoted to IEEE double; complex
   * values pass through unchanged.
   */
  exprt normalize_numeric_expr(const exprt &value) const;

private:
  python_converter &converter_;
  contextt &symbol_table_;
  type_handler &type_handler_;

  /// Intra-call symbol cache (cleared at each public entry point).
  mutable std::unordered_map<std::string, const symbolt *> symbol_cache_;

  /// Clear the symbol cache at the beginning of each public method.
  void clear_cache() const;

  /// Cached symbol lookup.
  const symbolt *find_cached_symbol(const std::string &id) const;

  // ---- shared IEEE / complex arithmetic helpers ----

  /// Builds a binary IEEE-754 operation node (e.g. ieee_add, ieee_mul).
  static exprt ieee_binop(const irep_idt &id, const exprt &x, const exprt &y);

  /// Complex multiplication: (a+bi)(c+di) = (ac-bd)+(ad+bc)i.
  exprt complex_mul(const exprt &x, const exprt &y) const;

  /**
   * Complex division with runtime ZeroDivisionError guard.
   * Emits an if-then guard on current_block when the denominator is 0+0j.
   */
  exprt complex_div(
    const exprt &x,
    const exprt &y,
    const nlohmann::json &loc_source) const;

  /// Natural logarithm of a complex number: ln|z| + i·arg(z).
  exprt complex_log(const exprt &z, const nlohmann::json &loc_source) const;

  /// Complex exponential: e^(a+bi) = e^a (cos b + i sin b).
  exprt complex_exp(const exprt &z, const nlohmann::json &loc_source) const;

  /// Recursively promotes an integer arithmetic tree to IEEE double.
  exprt
  promote_int_arith_to_double(const exprt &input_expr, std::size_t depth) const;
};

#endif // PYTHON_FRONTEND_COMPLEX_HANDLER_H
