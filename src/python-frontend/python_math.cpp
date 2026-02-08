#include <python-frontend/python_math.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/convert_float_literal.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/std_code.h>
#include <util/message.h>

#include <cmath>

python_math::python_math(
  python_converter &conv,
  contextt &ctx,
  type_handler &th)
  : converter(conv), symbol_table(ctx), type_handler_(th)
{
}

exprt python_math::resolve_symbol(const exprt &operand) const
{
  if (operand.is_symbol())
  {
    symbolt *s = symbol_table.find_symbol(operand.identifier());
    assert(s && "Symbol not found in symbol table");
    return s->value;
  }
  return operand;
}

exprt python_math::compute_expr(const exprt &expr) const
{
  // Resolve operands to their constant values
  const exprt lhs = resolve_symbol(expr.operands().at(0));
  const exprt rhs = resolve_symbol(expr.operands().at(1));

  // Convert to BigInt for computation
  const BigInt op1 =
    binary2integer(lhs.value().as_string(), lhs.type().is_signedbv());
  const BigInt op2 =
    binary2integer(rhs.value().as_string(), rhs.type().is_signedbv());

  // Perform the math operation
  BigInt result;
  if (expr.id() == "+")
    result = op1 + op2;
  else if (expr.id() == "-")
    result = op1 - op2;
  else if (expr.id() == "*")
    result = op1 * op2;
  else if (expr.id() == "/")
    result = op1 / op2;
  else
    throw std::runtime_error("Unsupported math operation: " + expr.id_string());

  // Return the result as a constant expression
  return constant_exprt(result, lhs.type());
}

exprt python_math::handle_power_symbolic(exprt base, exprt exp)
{
  // Find the pow function symbol
  symbolt *pow_symbol = symbol_table.find_symbol("c:@F@pow");
  if (!pow_symbol)
    throw std::runtime_error("pow function not found in symbol table");

  // Convert arguments to double type if needed
  exprt double_base = base;
  exprt double_exp = exp;

  if (!base.type().is_floatbv())
  {
    double_base = exprt("typecast", double_type());
    double_base.copy_to_operands(base);
  }

  if (!exp.type().is_floatbv())
  {
    double_exp = exprt("typecast", double_type());
    double_exp.copy_to_operands(exp);
  }

  // Create the function call
  side_effect_expr_function_callt pow_call;
  pow_call.function() = symbol_expr(*pow_symbol);
  pow_call.arguments() = {double_base, double_exp};
  pow_call.type() = double_type();

  // Always return double result: Python power with float operands returns float
  return pow_call;
}

exprt python_math::build_power_expression(const exprt &base, const BigInt &exp)
{
  if (exp == 0)
    return from_integer(1, base.type());
  if (exp == 1)
    return base;

  // For small exponents, use simple multiplication chain
  if (exp <= 10)
  {
    exprt result = base;
    for (BigInt i = 1; i < exp; ++i)
    {
      exprt mul_expr("*", base.type());
      mul_expr.copy_to_operands(result, base);
      result = mul_expr;
    }
    return result;
  }

  // For larger exponents, use exponentiation by squaring
  // This reduces the number of operations from O(n) to O(log n)
  if (exp % 2 == 0)
  {
    // Even exponent: (base^2)^(exp/2)
    exprt square("*", base.type());
    square.copy_to_operands(base, base);
    return build_power_expression(square, exp / 2);
  }
  else
  {
    // Odd exponent: base * base^(exp-1)
    exprt mul_expr("*", base.type());
    exprt sub_power = build_power_expression(base, exp - 1);
    mul_expr.copy_to_operands(base, sub_power);
    return mul_expr;
  }
}

exprt python_math::handle_power(exprt lhs, exprt rhs)
{
  // Handle pow symbolically if one of the operands is floatbv
  if (lhs.type().is_floatbv() || rhs.type().is_floatbv())
    return handle_power_symbolic(lhs, rhs);

  // Helper lambda to check if expression is a math expression
  auto is_math_expr = [](const exprt &expr) {
    const std::string &id = expr.id().as_string();
    return id == "+" || id == "-" || id == "*" || id == "/";
  };

  // Try to resolve constant values of both lhs and rhs
  exprt resolved_lhs = lhs;
  if (lhs.is_symbol())
  {
    const symbolt *s = symbol_table.find_symbol(lhs.identifier());
    if (s && !s->value.value().empty())
      resolved_lhs = s->value;
  }
  else if (is_math_expr(lhs))
    resolved_lhs = compute_expr(lhs);

  exprt resolved_rhs = rhs;
  if (rhs.is_symbol())
  {
    const symbolt *s = symbol_table.find_symbol(rhs.identifier());
    if (s && !s->value.value().empty())
      resolved_rhs = s->value;
  }
  else if (is_math_expr(rhs))
    resolved_rhs = compute_expr(rhs);

  // If rhs is still not constant, we need to handle this case
  if (!resolved_rhs.is_constant())
  {
    log_warning(
      "ESBMC-Python does not support power expressions with non-constant "
      "exponents");
    return from_integer(1, lhs.type());
  }

  // Check if the exponent is a floating-point number
  if (resolved_rhs.type().is_floatbv())
  {
    log_warning("ESBMC-Python does not support floating-point exponents yet");
    return from_integer(1, lhs.type());
  }

  // Convert rhs to integer exponent
  BigInt exponent;
  try
  {
    exponent = binary2integer(
      resolved_rhs.value().as_string(), resolved_rhs.type().is_signedbv());
  }
  catch (...)
  {
    log_warning("Failed to convert exponent to integer");
    return from_integer(1, lhs.type());
  }

  // Handle negative exponents via pow() to preserve semantics
  if (exponent < 0)
  {
    log_warning(
      "Handling negative exponents via pow() and returning a floating-point "
      "value");
    return handle_power_symbolic(lhs, rhs);
  }

  // Handle special cases first
  if (exponent == 0)
    return from_integer(1, lhs.type());
  if (exponent == 1)
    return lhs;

  // Check resolved base for special cases
  if (resolved_lhs.is_constant())
  {
    BigInt base = binary2integer(
      resolved_lhs.value().as_string(), resolved_lhs.type().is_signedbv());

    // Special cases for constant base
    if (base == 0 && exponent > 0)
      return from_integer(0, lhs.type());
    if (base == 1)
      return from_integer(1, lhs.type());
    if (base == -1)
      return from_integer((exponent % 2 == 0) ? 1 : -1, lhs.type());
  }

  // Build symbolic multiplication tree using exponentiation by squaring for efficiency
  return build_power_expression(lhs, exponent);
}

exprt python_math::handle_modulo(
  exprt lhs,
  exprt rhs,
  const nlohmann::json &element)
{
  // Find required function symbols
  symbolt *floor_symbol = symbol_table.find_symbol("c:@F@floor");
  if (!floor_symbol)
    throw std::runtime_error("floor function not found in symbol table");

  // Promote both operands to double if needed
  exprt double_lhs = lhs;
  exprt double_rhs = rhs;

  if (!lhs.type().is_floatbv())
  {
    double_lhs = exprt("typecast", double_type());
    double_lhs.copy_to_operands(lhs);
  }

  if (!rhs.type().is_floatbv())
  {
    double_rhs = exprt("typecast", double_type());
    double_rhs.copy_to_operands(rhs);
  }

  // Create division: x / y
  exprt div_expr("ieee_div", double_type());
  div_expr.copy_to_operands(double_lhs, double_rhs);

  // Create floor(x / y)
  side_effect_expr_function_callt floor_call;
  floor_call.function() = symbol_expr(*floor_symbol);
  floor_call.arguments() = {div_expr};
  floor_call.type() = double_type();
  floor_call.location() = converter.get_location_from_decl(element);

  // Create floor(x/y) * y
  exprt mult_expr("ieee_mul", double_type());
  mult_expr.copy_to_operands(floor_call, double_rhs);

  // Create x - floor(x/y) * y
  exprt result_expr("ieee_sub", double_type());
  result_expr.copy_to_operands(double_lhs, mult_expr);
  result_expr.location() = converter.get_location_from_decl(element);

  return result_expr;
}

exprt python_math::handle_floor_division(
  const exprt &lhs,
  const exprt &rhs,
  const exprt &bin_expr)
{
  typet div_type = bin_expr.type();

  // remainder = num % den;
  exprt remainder("mod", div_type);
  remainder.copy_to_operands(lhs, rhs);

  // Get num signal
  exprt is_num_neg("<", bool_type());
  is_num_neg.copy_to_operands(lhs, gen_zero(div_type));

  // Get den signal
  exprt is_den_neg("<", bool_type());
  is_den_neg.copy_to_operands(rhs, gen_zero(div_type));

  // remainder != 0
  exprt pos_remainder("notequal", bool_type());
  pos_remainder.copy_to_operands(remainder, gen_zero(div_type));

  // diff_signals = is_num_neg ^ is_den_neg;
  exprt diff_signals("bitxor", bool_type());
  diff_signals.copy_to_operands(is_num_neg, is_den_neg);

  exprt cond("and", bool_type());
  cond.copy_to_operands(pos_remainder, diff_signals);

  exprt if_expr("if", div_type);
  if_expr.copy_to_operands(cond, gen_one(div_type), gen_zero(div_type));

  // floor_div = (lhs / rhs) - (1 if (lhs % rhs != 0) and (lhs < 0) ^ (rhs < 0) else 0)
  exprt floor_div("-", div_type);
  floor_div.copy_to_operands(bin_expr, if_expr); // bin_expr contains lhs/rhs

  return floor_div;
}

void python_math::handle_float_division(exprt &lhs, exprt &rhs, exprt &bin_expr)
  const
{
  const typet float_type = double_type();

  auto promote_to_float = [&](exprt &e) {
    const typet &t = e.type();
    const bool is_integer = type_utils::is_integer_type(t);

    if (!is_integer)
      return;

    // Handle constant integers: convert them to float literals
    if (e.is_constant())
    {
      try
      {
        const bool is_signed = t.is_signedbv();
        const BigInt val = binary2integer(e.value().as_string(), is_signed);
        const double float_val = static_cast<double>(val.to_int64());
        convert_float_literal(std::to_string(float_val), e);
      }
      catch (const std::exception &ex)
      {
        log_error(
          "handle_float_division: failed to promote constant to float: {}",
          ex.what());
      }
    }
    else
    {
      // For non-constant operands (like function parameters), create explicit typecast expression
      e = typecast_exprt(e, float_type);
    }
  };

  promote_to_float(lhs);
  promote_to_float(rhs);

  // Set the result type and operator ID to reflect float division
  bin_expr.type() = float_type;
  bin_expr.id("ieee_div");
}

void python_math::promote_int_to_float(exprt &op, const typet &target_type)
  const
{
  typet &op_type = op.type();

  // Only promote if operand is an integer type
  if (!(type_utils::is_integer_type(op_type)))
    return;

  // Handle constant integers
  if (op.is_constant())
  {
    try
    {
      const BigInt int_val =
        binary2integer(op.value().as_string(), op_type.is_signedbv());

      // Generate a string like "3.0" for float parsing
      const std::string float_literal =
        std::to_string(int_val.to_int64()) + ".0";

      // Convert string literal to float expression
      convert_float_literal(float_literal, op);
    }
    catch (const std::exception &e)
    {
      log_error(
        "math_handler_.promote_int_to_float: Failed to promote constant to "
        "float: {}",
        e.what());
      return;
    }
  }

  // Update the operand type
  op.type() = target_type;

  // Update symbol type info if necessary
  if (op.is_symbol())
    converter.update_symbol(op);
}

exprt python_math::handle_sqrt(exprt operand, const nlohmann::json &element)
{
  // Find the sqrt function symbol from C math library
  symbolt *sqrt_symbol = symbol_table.find_symbol("c:@F@sqrt");
  if (!sqrt_symbol)
    throw std::runtime_error("sqrt function not found in symbol table");

  // Promote operand to double if needed (sqrt always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt sqrt_call;
  sqrt_call.function() = symbol_expr(*sqrt_symbol);
  sqrt_call.arguments() = {double_operand};
  sqrt_call.type() = double_type();
  sqrt_call.location() = converter.get_location_from_decl(element);

  return sqrt_call;
}

exprt python_math::handle_divmod(
  exprt dividend,
  exprt divisor,
  const nlohmann::json &element)
{
  // Determine the result type (promote to float if either operand is float)
  typet result_type = dividend.type();
  if (dividend.type().is_floatbv() || divisor.type().is_floatbv())
  {
    result_type = double_type();

    // Promote operands to float if needed
    if (!dividend.type().is_floatbv())
    {
      exprt promoted = exprt("typecast", result_type);
      promoted.copy_to_operands(dividend);
      dividend = promoted;
    }
    if (!divisor.type().is_floatbv())
    {
      exprt promoted = exprt("typecast", result_type);
      promoted.copy_to_operands(divisor);
      divisor = promoted;
    }
  }

  // Calculate quotient: a // b (floor division)
  exprt quotient;
  if (result_type.is_floatbv())
  {
    // For floats, use floor(a / b)
    exprt div_expr("ieee_div", result_type);
    div_expr.copy_to_operands(dividend, divisor);

    symbolt *floor_symbol = symbol_table.find_symbol("c:@F@floor");
    if (!floor_symbol)
      throw std::runtime_error("floor function not found in symbol table");

    side_effect_expr_function_callt floor_call;
    floor_call.function() = symbol_expr(*floor_symbol);
    floor_call.arguments() = {div_expr};
    floor_call.type() = result_type;
    floor_call.location() = converter.get_location_from_decl(element);
    quotient = floor_call;
  }
  else
  {
    // For integers, use integer division then apply floor division logic
    exprt int_div("/", result_type);
    int_div.copy_to_operands(dividend, divisor);
    quotient = handle_floor_division(dividend, divisor, int_div);
  }

  // Calculate remainder: a % b
  // Python's remainder: a - (a // b) * b
  // This ensures the remainder has the same sign as the divisor
  exprt remainder;
  if (result_type.is_floatbv())
  {
    remainder = handle_modulo(dividend, divisor, element);
  }
  else
  {
    // For integers, compute: remainder = dividend - quotient * divisor
    // This matches Python's semantics: divmod(a, b) = (q, r) where a = q*b + r
    exprt quotient_times_divisor("*", result_type);
    quotient_times_divisor.copy_to_operands(quotient, divisor);

    remainder = exprt("-", result_type);
    remainder.copy_to_operands(dividend, quotient_times_divisor);
  }

  // Create a tuple struct to hold both values
  struct_typet tuple_type;
  tuple_type.tag("tag-tuple_divmod");

  struct_typet::componentt comp0;
  comp0.name("element_0");
  comp0.type() = result_type;
  tuple_type.components().push_back(comp0);

  struct_typet::componentt comp1;
  comp1.name("element_1");
  comp1.type() = result_type;
  tuple_type.components().push_back(comp1);

  // Build the tuple expression
  exprt tuple_expr("struct", tuple_type);
  tuple_expr.copy_to_operands(quotient, remainder);

  return tuple_expr;
}

exprt python_math::handle_sin(exprt operand, const nlohmann::json &element)
{
  // Find the sin function symbol from C math library
  symbolt *sin_symbol = symbol_table.find_symbol("c:@F@sin");
  if (!sin_symbol)
    throw std::runtime_error("sin function not found in symbol table");

  // Promote operand to double if needed (sin always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt sin_call;
  sin_call.function() = symbol_expr(*sin_symbol);
  sin_call.arguments() = {double_operand};
  sin_call.type() = double_type();
  sin_call.location() = converter.get_location_from_decl(element);

  return sin_call;
}

exprt python_math::handle_cos(exprt operand, const nlohmann::json &element)
{
  // Find the cos function symbol from C math library
  symbolt *cos_symbol = symbol_table.find_symbol("c:@F@cos");
  if (!cos_symbol)
    throw std::runtime_error("cos function not found in symbol table");

  // Promote operand to double if needed (cos always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt cos_call;
  cos_call.function() = symbol_expr(*cos_symbol);
  cos_call.arguments() = {double_operand};
  cos_call.type() = double_type();
  cos_call.location() = converter.get_location_from_decl(element);

  return cos_call;
}
