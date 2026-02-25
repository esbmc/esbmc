#include <python-frontend/python_math.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/convert_float_literal.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/std_code.h>
#include <util/std_types.h>
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

exprt python_math::handle_exp(exprt operand, const nlohmann::json &element)
{
  // Find the exp function symbol from C math library
  symbolt *exp_symbol = symbol_table.find_symbol("c:@F@exp");
  if (!exp_symbol)
    throw std::runtime_error("exp function not found in symbol table");

  // Promote operand to double if needed (exp always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt exp_call;
  exp_call.function() = symbol_expr(*exp_symbol);
  exp_call.arguments() = {double_operand};
  exp_call.type() = double_type();
  exp_call.location() = converter.get_location_from_decl(element);

  return exp_call;
}

exprt python_math::handle_log(exprt operand, const nlohmann::json &element)
{
  // Find the log function symbol from C math library
  symbolt *log_symbol = symbol_table.find_symbol("c:@F@log");
  if (!log_symbol)
    throw std::runtime_error("log function not found in symbol table");

  // Promote operand to double if needed (log always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt log_call;
  log_call.function() = symbol_expr(*log_symbol);
  log_call.arguments() = {double_operand};
  log_call.type() = double_type();
  log_call.location() = converter.get_location_from_decl(element);

  return log_call;
}

exprt python_math::handle_acos(exprt operand, const nlohmann::json &element)
{
  // Find the acos function symbol from C math library
  symbolt *acos_symbol = symbol_table.find_symbol("c:@F@acos");
  if (!acos_symbol)
    throw std::runtime_error("acos function not found in symbol table");

  // Promote operand to double if needed (acos always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt acos_call;
  acos_call.function() = symbol_expr(*acos_symbol);
  acos_call.arguments() = {double_operand};
  acos_call.type() = double_type();
  acos_call.location() = converter.get_location_from_decl(element);

  return acos_call;
}

exprt python_math::handle_atan(exprt operand, const nlohmann::json &element)
{
  // Find the atan function symbol from C math library
  symbolt *atan_symbol = symbol_table.find_symbol("c:@F@atan");
  if (!atan_symbol)
    throw std::runtime_error("atan function not found in symbol table");

  // Promote operand to double if needed (atan always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt atan_call;
  atan_call.function() = symbol_expr(*atan_symbol);
  atan_call.arguments() = {double_operand};
  atan_call.type() = double_type();
  atan_call.location() = converter.get_location_from_decl(element);

  return atan_call;
}

exprt python_math::handle_atan2(
  exprt y_operand,
  exprt x_operand,
  const nlohmann::json &element)
{
  // Find the atan2 function symbol from C math library
  symbolt *atan2_symbol = symbol_table.find_symbol("c:@F@atan2");
  if (!atan2_symbol)
    throw std::runtime_error("atan2 function not found in symbol table");

  // Promote operands to double if needed (atan2 always works with doubles)
  if (!y_operand.type().is_floatbv())
  {
    exprt casted = exprt("typecast", double_type());
    casted.copy_to_operands(y_operand);
    y_operand = casted;
  }
  if (!x_operand.type().is_floatbv())
  {
    exprt casted = exprt("typecast", double_type());
    casted.copy_to_operands(x_operand);
    x_operand = casted;
  }

  // Create the function call expression
  side_effect_expr_function_callt atan2_call;
  atan2_call.function() = symbol_expr(*atan2_symbol);
  atan2_call.arguments() = {y_operand, x_operand};
  atan2_call.type() = double_type();
  atan2_call.location() = converter.get_location_from_decl(element);

  return atan2_call;
}

exprt python_math::handle_log2(exprt operand, const nlohmann::json &element)
{
  // Find the log2 function symbol from C math library
  symbolt *log2_symbol = symbol_table.find_symbol("c:@F@log2");
  if (!log2_symbol)
    throw std::runtime_error("log2 function not found in symbol table");

  // Promote operand to double if needed (log2 always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt log2_call;
  log2_call.function() = symbol_expr(*log2_symbol);
  log2_call.arguments() = {double_operand};
  log2_call.type() = double_type();
  log2_call.location() = converter.get_location_from_decl(element);

  return log2_call;
}

exprt python_math::handle_pow(
  exprt base,
  exprt exp,
  const nlohmann::json &element)
{
  // Find the pow function symbol from C math library
  symbolt *pow_symbol = symbol_table.find_symbol("c:@F@pow");
  if (!pow_symbol)
    throw std::runtime_error("pow function not found in symbol table");

  // Promote operands to double if needed (pow always works with doubles)
  if (!base.type().is_floatbv())
  {
    exprt casted = exprt("typecast", double_type());
    casted.copy_to_operands(base);
    base = casted;
  }
  if (!exp.type().is_floatbv())
  {
    exprt casted = exprt("typecast", double_type());
    casted.copy_to_operands(exp);
    exp = casted;
  }

  // Create the function call expression
  side_effect_expr_function_callt pow_call;
  pow_call.function() = symbol_expr(*pow_symbol);
  pow_call.arguments() = {base, exp};
  pow_call.type() = double_type();
  pow_call.location() = converter.get_location_from_decl(element);

  return pow_call;
}

exprt python_math::handle_fabs(exprt operand, const nlohmann::json &element)
{
  // Find the fabs function symbol from C math library
  symbolt *fabs_symbol = symbol_table.find_symbol("c:@F@fabs");
  if (!fabs_symbol)
    throw std::runtime_error("fabs function not found in symbol table");

  // Promote operand to double if needed (fabs always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt fabs_call;
  fabs_call.function() = symbol_expr(*fabs_symbol);
  fabs_call.arguments() = {double_operand};
  fabs_call.type() = double_type();
  fabs_call.location() = converter.get_location_from_decl(element);

  return fabs_call;
}

exprt python_math::handle_trunc(exprt operand, const nlohmann::json &element)
{
  // Find the trunc function symbol from C math library
  symbolt *trunc_symbol = symbol_table.find_symbol("c:@F@trunc");
  if (!trunc_symbol)
    throw std::runtime_error("trunc function not found in symbol table");

  // Promote operand to double if needed (trunc always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt trunc_call;
  trunc_call.function() = symbol_expr(*trunc_symbol);
  trunc_call.arguments() = {double_operand};
  trunc_call.type() = double_type();
  trunc_call.location() = converter.get_location_from_decl(element);

  exprt to_int("typecast", int_type());
  to_int.copy_to_operands(trunc_call);
  return to_int;
}

exprt python_math::handle_fmod(
  exprt lhs,
  exprt rhs,
  const nlohmann::json &element)
{
  // Find the fmod function symbol from C math library
  symbolt *fmod_symbol = symbol_table.find_symbol("c:@F@fmod");
  if (!fmod_symbol)
    throw std::runtime_error("fmod function not found in symbol table");

  // Promote operands to double if needed (fmod always works with doubles)
  if (!lhs.type().is_floatbv())
  {
    exprt casted = exprt("typecast", double_type());
    casted.copy_to_operands(lhs);
    lhs = casted;
  }
  if (!rhs.type().is_floatbv())
  {
    exprt casted = exprt("typecast", double_type());
    casted.copy_to_operands(rhs);
    rhs = casted;
  }

  // Create the function call expression
  side_effect_expr_function_callt fmod_call;
  fmod_call.function() = symbol_expr(*fmod_symbol);
  fmod_call.arguments() = {lhs, rhs};
  fmod_call.type() = double_type();
  fmod_call.location() = converter.get_location_from_decl(element);

  return fmod_call;
}

exprt python_math::handle_copysign(
  exprt lhs,
  exprt rhs,
  const nlohmann::json &element)
{
  // Find the copysign function symbol from C math library
  symbolt *copysign_symbol = symbol_table.find_symbol("c:@F@copysign");
  if (!copysign_symbol)
    throw std::runtime_error("copysign function not found in symbol table");

  // Promote operands to double if needed (copysign always works with doubles)
  if (!lhs.type().is_floatbv())
  {
    exprt casted = exprt("typecast", double_type());
    casted.copy_to_operands(lhs);
    lhs = casted;
  }
  if (!rhs.type().is_floatbv())
  {
    exprt casted = exprt("typecast", double_type());
    casted.copy_to_operands(rhs);
    rhs = casted;
  }

  // Create the function call expression
  side_effect_expr_function_callt copysign_call;
  copysign_call.function() = symbol_expr(*copysign_symbol);
  copysign_call.arguments() = {lhs, rhs};
  copysign_call.type() = double_type();
  copysign_call.location() = converter.get_location_from_decl(element);

  return copysign_call;
}

exprt python_math::handle_tan(exprt operand, const nlohmann::json &element)
{
  // Find the tan function symbol from C math library
  symbolt *tan_symbol = symbol_table.find_symbol("c:@F@tan");
  if (!tan_symbol)
    throw std::runtime_error("tan function not found in symbol table");

  // Promote operand to double if needed (tan always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt tan_call;
  tan_call.function() = symbol_expr(*tan_symbol);
  tan_call.arguments() = {double_operand};
  tan_call.type() = double_type();
  tan_call.location() = converter.get_location_from_decl(element);

  return tan_call;
}

exprt python_math::handle_asin(exprt operand, const nlohmann::json &element)
{
  // Find the asin function symbol from C math library
  symbolt *asin_symbol = symbol_table.find_symbol("c:@F@asin");
  if (!asin_symbol)
    throw std::runtime_error("asin function not found in symbol table");

  // Promote operand to double if needed (asin always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt asin_call;
  asin_call.function() = symbol_expr(*asin_symbol);
  asin_call.arguments() = {double_operand};
  asin_call.type() = double_type();
  asin_call.location() = converter.get_location_from_decl(element);

  return asin_call;
}

exprt python_math::handle_sinh(exprt operand, const nlohmann::json &element)
{
  // Find the sinh function symbol from C math library
  symbolt *sinh_symbol = symbol_table.find_symbol("c:@F@sinh");
  if (!sinh_symbol)
    throw std::runtime_error("sinh function not found in symbol table");

  // Promote operand to double if needed (sinh always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt sinh_call;
  sinh_call.function() = symbol_expr(*sinh_symbol);
  sinh_call.arguments() = {double_operand};
  sinh_call.type() = double_type();
  sinh_call.location() = converter.get_location_from_decl(element);

  return sinh_call;
}

exprt python_math::handle_cosh(exprt operand, const nlohmann::json &element)
{
  // Find the cosh function symbol from C math library
  symbolt *cosh_symbol = symbol_table.find_symbol("c:@F@cosh");
  if (!cosh_symbol)
    throw std::runtime_error("cosh function not found in symbol table");

  // Promote operand to double if needed (cosh always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt cosh_call;
  cosh_call.function() = symbol_expr(*cosh_symbol);
  cosh_call.arguments() = {double_operand};
  cosh_call.type() = double_type();
  cosh_call.location() = converter.get_location_from_decl(element);

  return cosh_call;
}

exprt python_math::handle_tanh(exprt operand, const nlohmann::json &element)
{
  // Find the tanh function symbol from C math library
  symbolt *tanh_symbol = symbol_table.find_symbol("c:@F@tanh");
  if (!tanh_symbol)
    throw std::runtime_error("tanh function not found in symbol table");

  // Promote operand to double if needed (tanh always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt tanh_call;
  tanh_call.function() = symbol_expr(*tanh_symbol);
  tanh_call.arguments() = {double_operand};
  tanh_call.type() = double_type();
  tanh_call.location() = converter.get_location_from_decl(element);

  return tanh_call;
}

exprt python_math::handle_log10(exprt operand, const nlohmann::json &element)
{
  // Find the log10 function symbol from C math library
  symbolt *log10_symbol = symbol_table.find_symbol("c:@F@log10");
  if (!log10_symbol)
    throw std::runtime_error("log10 function not found in symbol table");

  // Promote operand to double if needed (log10 always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt log10_call;
  log10_call.function() = symbol_expr(*log10_symbol);
  log10_call.arguments() = {double_operand};
  log10_call.type() = double_type();
  log10_call.location() = converter.get_location_from_decl(element);

  return log10_call;
}

exprt python_math::handle_expm1(exprt operand, const nlohmann::json &element)
{
  // Find the expm1 function symbol from C math library
  symbolt *expm1_symbol = symbol_table.find_symbol("c:@F@expm1");
  if (!expm1_symbol)
    throw std::runtime_error("expm1 function not found in symbol table");

  // Promote operand to double if needed (expm1 always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt expm1_call;
  expm1_call.function() = symbol_expr(*expm1_symbol);
  expm1_call.arguments() = {double_operand};
  expm1_call.type() = double_type();
  expm1_call.location() = converter.get_location_from_decl(element);

  return expm1_call;
}

exprt python_math::handle_log1p(exprt operand, const nlohmann::json &element)
{
  // Find the log1p function symbol from C math library
  symbolt *log1p_symbol = symbol_table.find_symbol("c:@F@log1p");
  if (!log1p_symbol)
    throw std::runtime_error("log1p function not found in symbol table");

  // Promote operand to double if needed (log1p always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt log1p_call;
  log1p_call.function() = symbol_expr(*log1p_symbol);
  log1p_call.arguments() = {double_operand};
  log1p_call.type() = double_type();
  log1p_call.location() = converter.get_location_from_decl(element);

  return log1p_call;
}

exprt python_math::handle_exp2(exprt operand, const nlohmann::json &element)
{
  // Find the exp2 function symbol from C math library
  symbolt *exp2_symbol = symbol_table.find_symbol("c:@F@exp2");
  if (!exp2_symbol)
    throw std::runtime_error("exp2 function not found in symbol table");

  // Promote operand to double if needed (exp2 always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt exp2_call;
  exp2_call.function() = symbol_expr(*exp2_symbol);
  exp2_call.arguments() = {double_operand};
  exp2_call.type() = double_type();
  exp2_call.location() = converter.get_location_from_decl(element);

  return exp2_call;
}

exprt python_math::handle_asinh(exprt operand, const nlohmann::json &element)
{
  // Find the asinh function symbol from C math library
  symbolt *asinh_symbol = symbol_table.find_symbol("c:@F@asinh");
  if (!asinh_symbol)
    throw std::runtime_error("asinh function not found in symbol table");

  // Promote operand to double if needed (asinh always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt asinh_call;
  asinh_call.function() = symbol_expr(*asinh_symbol);
  asinh_call.arguments() = {double_operand};
  asinh_call.type() = double_type();
  asinh_call.location() = converter.get_location_from_decl(element);

  return asinh_call;
}

exprt python_math::handle_acosh(exprt operand, const nlohmann::json &element)
{
  // Find the acosh function symbol from C math library
  symbolt *acosh_symbol = symbol_table.find_symbol("c:@F@acosh");
  if (!acosh_symbol)
    throw std::runtime_error("acosh function not found in symbol table");

  // Promote operand to double if needed (acosh always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt acosh_call;
  acosh_call.function() = symbol_expr(*acosh_symbol);
  acosh_call.arguments() = {double_operand};
  acosh_call.type() = double_type();
  acosh_call.location() = converter.get_location_from_decl(element);

  return acosh_call;
}

exprt python_math::handle_atanh(exprt operand, const nlohmann::json &element)
{
  // Find the atanh function symbol from C math library
  symbolt *atanh_symbol = symbol_table.find_symbol("c:@F@atanh");
  if (!atanh_symbol)
    throw std::runtime_error("atanh function not found in symbol table");

  // Promote operand to double if needed (atanh always works with doubles)
  exprt double_operand = operand;
  if (!operand.type().is_floatbv())
  {
    double_operand = exprt("typecast", double_type());
    double_operand.copy_to_operands(operand);
  }

  // Create the function call expression
  side_effect_expr_function_callt atanh_call;
  atanh_call.function() = symbol_expr(*atanh_symbol);
  atanh_call.arguments() = {double_operand};
  atanh_call.type() = double_type();
  atanh_call.location() = converter.get_location_from_decl(element);

  return atanh_call;
}

exprt python_math::handle_hypot(
  exprt lhs,
  exprt rhs,
  const nlohmann::json &element)
{
  // Find the hypot function symbol from C math library
  symbolt *hypot_symbol = symbol_table.find_symbol("c:@F@hypot");
  if (!hypot_symbol)
    throw std::runtime_error("hypot function not found in symbol table");

  // Promote operands to double if needed (hypot always works with doubles)
  if (!lhs.type().is_floatbv())
  {
    exprt casted = exprt("typecast", double_type());
    casted.copy_to_operands(lhs);
    lhs = casted;
  }
  if (!rhs.type().is_floatbv())
  {
    exprt casted = exprt("typecast", double_type());
    casted.copy_to_operands(rhs);
    rhs = casted;
  }

  // Create the function call expression
  side_effect_expr_function_callt hypot_call;
  hypot_call.function() = symbol_expr(*hypot_symbol);
  hypot_call.arguments() = {lhs, rhs};
  hypot_call.type() = double_type();
  hypot_call.location() = converter.get_location_from_decl(element);

  return hypot_call;
}

exprt python_math::handle_dist(
  exprt p,
  exprt q,
  const nlohmann::json &element)
{
  // Both arguments must be tuples (struct types with tag-tuple prefix)
  if (!p.type().is_struct() || !q.type().is_struct())
    throw std::runtime_error("math.dist() arguments must be tuples");

  const struct_typet &p_type = to_struct_type(p.type());
  const struct_typet &q_type = to_struct_type(q.type());

  if (
    p_type.tag().as_string().find("tag-tuple") != 0 ||
    q_type.tag().as_string().find("tag-tuple") != 0)
    throw std::runtime_error("math.dist() arguments must be tuples");

  size_t p_size = p_type.components().size();
  size_t q_size = q_type.components().size();

  if (p_size != q_size)
    throw std::runtime_error(
      "math.dist() requires both points to have the same number of dimensions");

  if (p_size == 0)
    throw std::runtime_error("math.dist() requires non-empty points");

  // Build sum of squared differences: (p[0]-q[0])^2 + (p[1]-q[1])^2 + ...
  // Use ieee_* operations for floating-point arithmetic
  exprt rounding_mode =
    symbol_exprt("c:@__ESBMC_rounding_mode", int_type());

  exprt total = exprt();
  for (size_t i = 0; i < p_size; i++)
  {
    std::string member_name = "element_" + std::to_string(i);
    const typet &comp_type = p_type.components()[i].type();

    exprt pi = member_exprt(p, member_name, comp_type);
    exprt qi = member_exprt(q, member_name, q_type.components()[i].type());

    // Cast to double if needed
    if (!pi.type().is_floatbv())
    {
      exprt casted = exprt("typecast", double_type());
      casted.copy_to_operands(pi);
      pi = casted;
    }
    if (!qi.type().is_floatbv())
    {
      exprt casted = exprt("typecast", double_type());
      casted.copy_to_operands(qi);
      qi = casted;
    }

    // d = p[i] - q[i]
    exprt diff("ieee_sub", double_type());
    diff.copy_to_operands(pi, qi);
    diff.add("rounding_mode") = rounding_mode;

    // d * d
    exprt sq("ieee_mul", double_type());
    sq.copy_to_operands(diff, diff);
    sq.add("rounding_mode") = rounding_mode;

    if (i == 0)
      total = sq;
    else
    {
      exprt sum("ieee_add", double_type());
      sum.copy_to_operands(total, sq);
      sum.add("rounding_mode") = rounding_mode;
      total = sum;
    }
  }

  // Return sqrt(total) using ieee_sqrt (SMT-level intrinsic)
  exprt sqrt_expr("ieee_sqrt", double_type());
  sqrt_expr.copy_to_operands(total);
  sqrt_expr.add("rounding_mode") = rounding_mode;
  sqrt_expr.location() = converter.get_location_from_decl(element);

  return sqrt_expr;
}
