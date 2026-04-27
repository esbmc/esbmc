#ifndef PYTHON_FRONTEND_COMPLEX_HANDLER_UTILS_H
#define PYTHON_FRONTEND_COMPLEX_HANDLER_UTILS_H

#include <string>
#include <util/expr.h>

class python_converter;

namespace complex_utils
{
/**
 * Parses a Python complex literal string into (real, imag) components.
 * Handles all CPython-accepted formats: "1+2j", "(3-4.5j)", "2j", "1.5",
 * "+j", "-j", scientific notation like "1e3+2e-1j", etc.
 *
 * @param raw   Input string (may include whitespace and wrapping parentheses).
 * @param real_out  Output real part.
 * @param imag_out  Output imaginary part.
 * @return true on success, false if the string is malformed.
 */
bool parse_complex_string(
  const std::string &raw,
  double &real_out,
  double &imag_out);

/**
 * Parses a Python complex literal string into (real, imag) exprt constants
 * using convert_float_literal, ensuring bit-exact consistency with constants
 * produced from Python AST float literals (same cformat and binary value).
 *
 * @param raw       Input string (may include whitespace and parentheses).
 * @param real_out  Output real part as a constant_exprt.
 * @param imag_out  Output imaginary part as a constant_exprt.
 * @return true on success, false if the string is malformed.
 */
bool parse_complex_string(
  const std::string &raw,
  exprt &real_out,
  exprt &imag_out);

/**
 * Generates a TypeError expression for math functions that reject
 * complex arguments: "must be real number, not complex".
 */
exprt raise_math_real_type_error_expr(python_converter &converter);

/**
 * Generates a TypeError expression for functions that require an integer
 * but received a complex: "'complex' object cannot be interpreted as an integer".
 */
exprt raise_math_int_type_error_expr(python_converter &converter);

} // namespace complex_utils

#endif
