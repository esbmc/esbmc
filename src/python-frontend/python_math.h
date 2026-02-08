#ifndef CPROVER_PYTHON_FRONTEND_PYTHON_MATH_H
#define CPROVER_PYTHON_FRONTEND_PYTHON_MATH_H

#include <util/arith_tools.h>
#include <util/expr.h>
#include <util/std_code.h>
#include <nlohmann/json.hpp>

class python_converter;
class type_handler;
class contextt;

/**
 * @brief Handles mathematical operations for Python-to-C conversion
 * 
 * This class encapsulates all math-related operations in the ESBMC Python frontend,
 * including arithmetic evaluation, power operations, division variants (true division,
 * floor division, modulo), and type promotions required by Python's semantics.
 */
class python_math
{
private:
  python_converter &converter;
  contextt &symbol_table;
  type_handler &type_handler_;

  /**
   * @brief Resolve a symbol to its constant value if possible
   * @param operand The operand to resolve (may be symbol or constant)
   * @return The resolved constant expression
   */
  exprt resolve_symbol(const exprt &operand) const;

public:
  /**
   * @brief Constructor
   * @param conv Reference to the parent python_converter
   * @param ctx Reference to the symbol table context
   * @param th Reference to the type handler
   */
  python_math(python_converter &conv, contextt &ctx, type_handler &th);

  /**
   * @brief Compute a mathematical expression with constant operands
   * 
   * Evaluates binary arithmetic operations (+, -, *, /) when both operands
   * are constants or can be resolved to constants via symbol lookup.
   * 
   * @param expr The binary expression to evaluate
   * @return A constant expression with the computed result
   * @throws std::runtime_error if operation is unsupported
   */
  exprt compute_expr(const exprt &expr) const;

  /**
   * @brief Handle Python's power operator (**)
   * 
   * Implements Python's exponentiation with special handling for:
   * - Floating-point operands (delegates to symbolic pow)
   * - Constant integer exponents (compile-time evaluation)
   * - Negative exponents (converts to symbolic)
   * - Special cases (0**n, 1**n, (-1)**n)
   * 
   * @param lhs Base expression
   * @param rhs Exponent expression
   * @return Expression representing base**exponent
   */
  exprt handle_power(exprt lhs, exprt rhs);

  /**
   * @brief Handle power operation symbolically using C's pow() function
   * 
   * Creates a function call to C's pow(base, exp) for cases where
   * compile-time evaluation isn't possible or operands are floating-point.
   * 
   * @param base Base expression (promoted to double if needed)
   * @param exp Exponent expression (promoted to double if needed)
   * @return Function call expression to pow()
   * @throws std::runtime_error if pow symbol not found
   */
  exprt handle_power_symbolic(exprt base, exprt exp);

  /**
   * @brief Build power expression using exponentiation by squaring
   * 
   * Efficiently constructs a multiplication tree for base**exp using
   * the binary exponentiation algorithm, reducing operations from O(n) to O(log n).
   * 
   * @param base The base expression
   * @param exp The exponent (must be non-negative integer)
   * @return Expression tree representing the power operation
   */
  exprt build_power_expression(const exprt &base, const BigInt &exp);

  /**
   * @brief Handle Python's modulo operator (%)
   * 
   * Implements Python's modulo semantics where the result has the sign of the divisor:
   * x % y = x - floor(x/y) * y
   * 
   * This differs from C's fmod where result has the sign of the dividend.
   * 
   * @param lhs Dividend expression
   * @param rhs Divisor expression
   * @param element JSON AST node for location information
   * @return Expression implementing Python modulo semantics
   * @throws std::runtime_error if floor function not found
   */
  exprt handle_modulo(exprt lhs, exprt rhs, const nlohmann::json &element);

  /**
   * @brief Handle Python's floor division operator (//)
   * 
   * Implements floor division: (lhs // rhs) = floor(lhs / rhs)
   * Special handling ensures correct behavior with negative operands:
   * result = (lhs / rhs) - (1 if (lhs % rhs != 0) and signs_differ else 0)
   * 
   * @param lhs Dividend expression
   * @param rhs Divisor expression
   * @param bin_expr The division expression to modify
   * @return Expression implementing floor division
   */
  exprt handle_floor_division(
    const exprt &lhs,
    const exprt &rhs,
    const exprt &bin_expr);

  /**
   * @brief Handle Python's true division operator (/)
   * 
   * Python 3's / operator always performs floating-point division,
   * even with integer operands. This function promotes operands to float
   * and sets up IEEE floating-point division.
   * 
   * @param lhs Left operand (modified to float if needed)
   * @param rhs Right operand (modified to float if needed)
   * @param bin_expr Binary expression (modified to ieee_div)
   */
  void handle_float_division(exprt &lhs, exprt &rhs, exprt &bin_expr) const;

  /**
   * @brief Promote integer expression to floating-point type
   * 
   * Converts integer operands to floatbv type, typically for Python's
   * true division where / must always yield float results.
   * 
   * Handles both constant integers (converts literal value) and
   * non-constant integers (updates type metadata).
   * 
   * @param op The operand to promote (modified in place)
   * @param target_type The target floating-point type
   */
  void promote_int_to_float(exprt &op, const typet &target_type) const;

  /**
   * @brief Handle square root operation (math.sqrt)
   * 
   * Implements Python's math.sqrt() function by creating a call to C's sqrt().
   * Always returns a float (double) result. Negative inputs will produce
   * domain errors at runtime (matching Python behavior).
   * 
   * @param operand The value to take square root of (promoted to float if needed)
   * @param element JSON AST node for location information (optional)
   * @return Function call expression to sqrt()
   * @throws std::runtime_error if sqrt symbol not found in symbol table
   * 
   * Examples:
   *   math.sqrt(4) -> 2.0
   *   math.sqrt(2) -> 1.414...
   *   math.sqrt(0) -> 0.0
   *   math.sqrt(-1) -> domain error at runtime
   */
  exprt handle_sqrt(exprt operand, const nlohmann::json &element);

  /**
   * @brief Handle sine function (math.sin)
   *
   * Implements Python's math.sin() function by creating a call to C's sin().
   * Always returns a float (double) result representing the sine of the angle
   * in radians.
   *
   * @param operand The angle in radians (promoted to float if needed)
   * @param element JSON AST node for location information
   * @return Function call expression to sin()
   * @throws std::runtime_error if sin symbol not found in symbol table
   *
   * Examples:
   *   math.sin(0) -> 0.0
   *   math.sin(math.pi/2) -> 1.0
   *   math.sin(math.pi) -> 0.0 (approximately)
   *   math.sin(-math.pi/2) -> -1.0
   */
  exprt handle_sin(exprt operand, const nlohmann::json &element);

  /**
   * @brief Handle cosine function (math.cos)
   *
   * Implements Python's math.cos() function by creating a call to C's cos().
   * Always returns a float (double) result representing the cosine of the angle
   * in radians.
   *
   * @param operand The angle in radians (promoted to float if needed)
   * @param element JSON AST node for location information
   * @return Function call expression to cos()
   * @throws std::runtime_error if cos symbol not found in symbol table
   *
   * Examples:
   *   math.cos(0) -> 1.0
   *   math.cos(math.pi/2) -> 0.0 (approximately)
   *   math.cos(math.pi) -> -1.0
   *   math.cos(2*math.pi) -> 1.0
   */
  exprt handle_cos(exprt operand, const nlohmann::json &element);

  /**
   * @brief Handle divmod() built-in function
   * 
   * Implements Python's divmod(a, b) function, which returns a tuple containing
   * the quotient and remainder of division: (a // b, a % b).
   * 
   * The function follows Python's floor division semantics:
   * - The quotient is computed using floor division (rounds toward negative infinity)
   * - The remainder has the same sign as the divisor
   * - The mathematical property always holds: a == (a // b) * b + (a % b)
   * 
   * Type handling:
   * - If either operand is float, both are promoted to float (double)
   * - Integer operands use integer floor division and modulo
   * - Float operands use C's floor() function and floating-point modulo
   * 
   * @param dividend The numerator (value to be divided)
   * @param divisor The denominator (value to divide by)
   * @param element JSON AST node for location information
   * @return Struct expression representing tuple (quotient, remainder) with
   *         components named "element_0" (quotient) and "element_1" (remainder)
   * @throws std::runtime_error if floor symbol not found (for float division)
   * 
   * Examples:
   *   divmod(7, 3)      -> (2, 1)      [7 = 3*2 + 1]
   *   divmod(-7, 3)     -> (-3, 2)     [-7 = 3*(-3) + 2]
   *   divmod(7, -3)     -> (-3, -2)    [7 = (-3)*(-3) + (-2)]
   *   divmod(-7, -3)    -> (2, -1)     [-7 = (-3)*2 + (-1)]
   *   divmod(7.5, 2.0)  -> (3.0, 1.5)  [7.5 = 2.0*3.0 + 1.5]
   *   divmod(10, 5)     -> (2, 0)      [exact division]
   */
  exprt
  handle_divmod(exprt dividend, exprt divisor, const nlohmann::json &element);
};

#endif
