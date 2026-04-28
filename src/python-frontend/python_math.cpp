#include <python-frontend/python_math.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/type_utils.h>
#include <python-frontend/math_guard_utils.h>
#include <util/arith_tools.h>
#include <util/c_types.h>
#include <util/ieee_float.h>
#include <util/std_code.h>
#include <util/std_types.h>
#include <util/message.h>

#include <cmath>
#include <limits>

namespace
{
const BigInt kMaxConstantFoldExponent = 1024;

std::string make_math_dispatch_cache_key(
  const std::string &caller,
  const std::string &func_name)
{
  if (caller.empty())
  {
    std::string key;
    key.reserve(8 + func_name.size()); // "global::"
    key += "global::";
    key += func_name;
    return key;
  }

  std::string key;
  key.reserve(6 + caller.size() + 2 + func_name.size()); // "attr::" + "::"
  key += "attr::";
  key += caller;
  key += "::";
  key += func_name;
  return key;
}

BigInt pow_bigint_non_negative(BigInt base, BigInt exp)
{
  BigInt result = 1;
  while (exp > 0)
  {
    if ((exp % 2) != 0)
      result *= base;
    exp /= 2;
    if (exp > 0)
      base *= base;
  }
  return result;
}

bool is_basic_math_expr(const exprt &expr)
{
  const irep_idt &id = expr.id();
  return id == "+" || id == "-" || id == "*" || id == "/";
}
} // namespace

python_math::python_math(
  python_converter &conv,
  contextt &ctx,
  type_handler &th)
  : converter(conv), symbol_table(ctx), type_handler_(th)
{
}

bool python_math::is_math_dispatch_target(
  const std::string &caller,
  const std::string &func_name) const
{
  if (func_name.empty())
    return false;

  const auto &math_module_names =
    math_guard_utils::math_module_function_names();
  const auto &math_wrapper_names =
    math_guard_utils::math_wrapper_function_names();
  if (caller != "math")
    return math_wrapper_names.count(func_name) != 0;

  if (math_module_names.count(func_name) != 0)
  {
    return true;
  }

  return math_wrapper_names.count(func_name) != 0;
}

bool python_math::is_math_dispatch_target_cached(
  const std::string &caller,
  const std::string &func_name)
{
  if (func_name.empty())
    return false;

  function_call_cache &cache = converter.get_function_call_cache();
  const std::string key = make_math_dispatch_cache_key(caller, func_name);
  if (std::optional<bool> cached = cache.get_math_dispatch_classification(key);
      cached.has_value())
  {
    return cached.value();
  }

  const bool matches = is_math_dispatch_target(caller, func_name);
  cache.set_math_dispatch_classification(key, matches);
  return matches;
}

bool python_math::is_unary_dispatch_function(std::string_view func_name) const
{
  return func_name == "sin" || func_name == "cos" || func_name == "exp" ||
         func_name == "atan" || func_name == "log2" || func_name == "tan" ||
         func_name == "asin" || func_name == "sinh" || func_name == "cosh" ||
         func_name == "tanh" || func_name == "log10" || func_name == "expm1" ||
         func_name == "log1p" || func_name == "exp2" || func_name == "asinh" ||
         func_name == "acosh" || func_name == "atanh" || func_name == "fabs" ||
         func_name == "trunc";
}

bool python_math::is_binary_dispatch_function(std::string_view func_name) const
{
  return func_name == "atan2" || func_name == "pow" || func_name == "fmod" ||
         func_name == "copysign" || func_name == "hypot";
}

exprt python_math::handle(
  std::string_view func_name,
  exprt operand,
  const nlohmann::json &element)
{
  if (func_name == "sin")
    return handle_sin(std::move(operand), element);
  if (func_name == "cos")
    return handle_cos(std::move(operand), element);
  if (func_name == "exp")
    return handle_exp(std::move(operand), element);
  if (func_name == "atan")
    return handle_atan(std::move(operand), element);
  if (func_name == "log2")
    return handle_log2(std::move(operand), element);
  if (func_name == "tan")
    return handle_tan(std::move(operand), element);
  if (func_name == "asin")
    return handle_asin(std::move(operand), element);
  if (func_name == "sinh")
    return handle_sinh(std::move(operand), element);
  if (func_name == "cosh")
    return handle_cosh(std::move(operand), element);
  if (func_name == "tanh")
    return handle_tanh(std::move(operand), element);
  if (func_name == "log10")
    return handle_log10(std::move(operand), element);
  if (func_name == "expm1")
    return handle_expm1(std::move(operand), element);
  if (func_name == "log1p")
    return handle_log1p(std::move(operand), element);
  if (func_name == "exp2")
    return handle_exp2(std::move(operand), element);
  if (func_name == "asinh")
    return handle_asinh(std::move(operand), element);
  if (func_name == "acosh")
    return handle_acosh(std::move(operand), element);
  if (func_name == "atanh")
    return handle_atanh(std::move(operand), element);
  if (func_name == "fabs")
    return handle_fabs(std::move(operand), element);
  if (func_name == "trunc")
    return handle_trunc(std::move(operand), element);

  return nil_exprt();
}

exprt python_math::handle(
  std::string_view func_name,
  exprt lhs,
  exprt rhs,
  const nlohmann::json &element)
{
  if (func_name == "atan2")
    return handle_atan2(std::move(lhs), std::move(rhs), element);
  if (func_name == "pow")
    return handle_pow(std::move(lhs), std::move(rhs), element);
  if (func_name == "fmod")
    return handle_fmod(std::move(lhs), std::move(rhs), element);
  if (func_name == "copysign")
    return handle_copysign(std::move(lhs), std::move(rhs), element);
  if (func_name == "hypot")
    return handle_hypot(std::move(lhs), std::move(rhs), element);

  return nil_exprt();
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

std::optional<double>
python_math::try_resolve_constant_double(const exprt &operand) const
{
  // Do not fold mutable symbols: symbol table values may be stale between
  // assignments (e.g. in chained compound assignments).
  if (operand.is_symbol())
    return std::nullopt;

  exprt resolved = operand;
  if (!resolved.is_constant())
    return std::nullopt;

  if (resolved.type().is_floatbv())
  {
    ieee_floatt f;
    f.spec = to_floatbv_type(resolved.type());
    f.unpack(binary2integer(resolved.value().as_string(), false));
    return f.to_double();
  }

  if (resolved.type().is_signedbv() || resolved.type().is_unsignedbv())
  {
    const BigInt int_val = binary2integer(
      resolved.value().as_string(), resolved.type().is_signedbv());
    if (resolved.type().is_unsignedbv())
    {
      if (!int_val.is_uint64())
        return std::nullopt;
      return static_cast<double>(int_val.to_uint64());
    }
    if (!int_val.is_int64())
      return std::nullopt;
    return static_cast<double>(int_val.to_int64());
  }

  return std::nullopt;
}

exprt python_math::promote_to_double_if_needed(exprt operand) const
{
  if (operand.type().is_floatbv())
    return operand;

  // Fast path: fold numeric constants directly to double literal to avoid
  // generating extra typecast IR in hot math call paths.
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value())
  {
    return from_double(*val, double_type());
  }

  exprt double_operand = exprt("typecast", double_type());
  double_operand.copy_to_operands(operand);
  return double_operand;
}

exprt python_math::build_unary_c_math_call(
  const char *symbol_id,
  const char *display_name,
  exprt operand,
  const nlohmann::json &element)
{
  side_effect_expr_function_callt call;
  call.function() =
    symbol_expr(get_c_math_symbol_cached(symbol_id, display_name));
  call.arguments() = {promote_to_double_if_needed(std::move(operand))};
  call.type() = double_type();
  call.location() = converter.get_location_from_decl(element);
  return call;
}

const symbolt &python_math::get_c_math_symbol_cached(
  const char *symbol_id,
  const char *display_name)
{
  const std::string_view symbol_key{symbol_id};
  if (auto it = c_math_symbol_cache_.find(symbol_key);
      it != c_math_symbol_cache_.end() && it->second != nullptr)
  {
    return *it->second;
  }

  symbolt *symbol = symbol_table.find_symbol(symbol_id);
  if (!symbol)
    throw std::runtime_error(
      std::string(display_name) + " function not found in symbol table");

  c_math_symbol_cache_[symbol_key] = symbol;
  return *symbol;
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
  // Convert arguments to double type if needed
  exprt double_base = promote_to_double_if_needed(std::move(base));
  exprt double_exp = promote_to_double_if_needed(std::move(exp));

  // Create the function call
  side_effect_expr_function_callt pow_call;
  pow_call.function() =
    symbol_expr(get_c_math_symbol_cached("c:@F@pow", "pow"));
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

  // For larger exponents, use an iterative exponentiation-by-squaring loop.
  // This keeps O(log n) multiplications without recursive call overhead.
  std::optional<exprt> result;
  exprt factor = base;
  BigInt remaining = exp;

  while (remaining > 0)
  {
    if (remaining % 2 != 0)
    {
      if (!result.has_value())
      {
        result = factor;
      }
      else
      {
        exprt mul_expr("*", base.type());
        mul_expr.copy_to_operands(*result, factor);
        result = mul_expr;
      }
    }

    remaining /= 2;
    if (remaining > 0)
    {
      exprt square("*", base.type());
      square.copy_to_operands(factor, factor);
      factor = square;
    }
  }

  return result.has_value() ? *result : from_integer(1, base.type());
}

exprt python_math::handle_power(exprt lhs, exprt rhs)
{
  // Handle pow symbolically if one of the operands is floatbv
  if (lhs.type().is_floatbv() || rhs.type().is_floatbv())
    return handle_power_symbolic(lhs, rhs);

  // Resolve exponent first. If it is non-constant, we can return early without
  // spending time resolving the base.
  exprt resolved_rhs = rhs;
  if (is_basic_math_expr(rhs))
    resolved_rhs = compute_expr(rhs);

  // If rhs is still not constant or is a float, delegate to pow() for
  // soundness. This correctly handles symbolic exponents and negative exponents
  // (which Python's ** returns as float).
  if (!resolved_rhs.is_constant() || resolved_rhs.type().is_floatbv())
    return handle_power_symbolic(lhs, rhs);

  // Convert rhs to integer exponent
  BigInt exponent;
  try
  {
    exponent = binary2integer(
      resolved_rhs.value().as_string(), resolved_rhs.type().is_signedbv());
  }
  catch (...)
  {
    return handle_power_symbolic(lhs, rhs);
  }

  // Negative exponents: Python returns a float (e.g. 2**-1 == 0.5)
  if (exponent < 0)
    return handle_power_symbolic(lhs, rhs);

  // Fast-path exponents before attempting any base constant resolution.
  if (exponent == 0)
    return from_integer(1, lhs.type());
  if (exponent == 1)
    return lhs;

  exprt resolved_lhs = lhs;
  if (is_basic_math_expr(lhs))
    resolved_lhs = compute_expr(lhs);

  std::optional<BigInt> resolved_base_value;
  if (
    resolved_lhs.is_constant() &&
    (resolved_lhs.type().is_signedbv() || resolved_lhs.type().is_unsignedbv()))
  {
    try
    {
      resolved_base_value = binary2integer(
        resolved_lhs.value().as_string(), resolved_lhs.type().is_signedbv());
      // Constant folding very large integer powers can be more expensive than
      // keeping a logarithmic symbolic tree; cap it to keep conversion fast.
      if (exponent <= kMaxConstantFoldExponent)
      {
        const BigInt power_value =
          pow_bigint_non_negative(*resolved_base_value, exponent);
        return from_integer(power_value, lhs.type());
      }
    }
    catch (...)
    {
      // Fall back to symbolic encoding if constant conversion overflows/ fails.
    }
  }

  // Check resolved base for special cases
  if (resolved_base_value.has_value())
  {
    const BigInt &base = *resolved_base_value;

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
  if (std::optional<double> lhs_const = try_resolve_constant_double(lhs),
      rhs_const = try_resolve_constant_double(rhs);
      lhs_const.has_value() && rhs_const.has_value() && *rhs_const != 0.0)
  {
    const double q = std::floor(*lhs_const / *rhs_const);
    const double r = *lhs_const - (q * *rhs_const);
    return from_double(r, double_type());
  }

  // Promote both operands to double if needed
  exprt double_lhs = promote_to_double_if_needed(std::move(lhs));
  exprt double_rhs = promote_to_double_if_needed(std::move(rhs));

  // Create division: x / y
  exprt div_expr("ieee_div", double_type());
  div_expr.copy_to_operands(double_lhs, double_rhs);

  // Create floor(x / y)
  side_effect_expr_function_callt floor_call;
  floor_call.function() =
    symbol_expr(get_c_math_symbol_cached("c:@F@floor", "floor"));
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
  if (
    lhs.type().is_floatbv() || rhs.type().is_floatbv() ||
    bin_expr.type().is_floatbv())
  {
    if (std::optional<double> lhs_const = try_resolve_constant_double(lhs),
        rhs_const = try_resolve_constant_double(rhs);
        lhs_const.has_value() && rhs_const.has_value() && *rhs_const != 0.0)
      return from_double(std::floor(*lhs_const / *rhs_const), double_type());

    exprt div_expr("ieee_div", double_type());
    div_expr.copy_to_operands(
      promote_to_double_if_needed(lhs), promote_to_double_if_needed(rhs));

    side_effect_expr_function_callt floor_call;
    floor_call.function() =
      symbol_expr(get_c_math_symbol_cached("c:@F@floor", "floor"));
    floor_call.arguments() = {div_expr};
    floor_call.type() = double_type();
    floor_call.location() = bin_expr.location();
    return floor_call;
  }

  if (lhs.type().is_signedbv() || lhs.type().is_unsignedbv())
  {
    // Only fold direct constants; folding symbols can become unsound when
    // symbol-table values are not kept in sync after reassignment.
    if (
      lhs.is_constant() && rhs.is_constant() &&
      (rhs.type().is_signedbv() || rhs.type().is_unsignedbv()))
    {
      const BigInt lhs_val =
        binary2integer(lhs.value().as_string(), lhs.type().is_signedbv());
      const BigInt rhs_val =
        binary2integer(rhs.value().as_string(), rhs.type().is_signedbv());

      if (rhs_val != 0)
      {
        BigInt q = lhs_val / rhs_val;
        const BigInt rem = lhs_val % rhs_val;
        const bool sign_diff = (lhs_val < 0) != (rhs_val < 0);
        if (rem != 0 && sign_diff)
          q -= 1;
        return from_integer(q, bin_expr.type());
      }
    }
  }

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
        double float_val;
        if (is_signed)
        {
          if (!val.is_int64())
            throw std::runtime_error("integer constant out of int64 range");
          float_val = static_cast<double>(val.to_int64());
        }
        else
        {
          if (!val.is_uint64())
            throw std::runtime_error("integer constant out of uint64 range");
          float_val = static_cast<double>(val.to_uint64());
        }
        e = from_double(float_val, float_type);
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
      double as_double;
      if (op_type.is_signedbv())
      {
        if (!int_val.is_int64())
          throw std::runtime_error("integer constant out of int64 range");
        as_double = static_cast<double>(int_val.to_int64());
      }
      else
      {
        if (!int_val.is_uint64())
          throw std::runtime_error("integer constant out of uint64 range");
        as_double = static_cast<double>(int_val.to_uint64());
      }
      op = from_double(as_double, target_type);
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
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value())
  {
    // Constant-fold when we have a concrete value that is either:
    //  - non-negative finite, or
    //  - NaN or +/-infinity (IEEE-754 defines std::sqrt for these)
    if (*val >= 0.0 || std::isnan(*val) || std::isinf(*val))
      return from_double(std::sqrt(*val), double_type());
  }
  return build_unary_c_math_call(
    "c:@F@sqrt", "sqrt", std::move(operand), element);
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
    if (std::optional<double> lhs_const = try_resolve_constant_double(dividend),
        rhs_const = try_resolve_constant_double(divisor);
        lhs_const.has_value() && rhs_const.has_value() && *rhs_const != 0.0)
    {
      const double q = std::floor(*lhs_const / *rhs_const);
      const double r = *lhs_const - (q * *rhs_const);

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

      exprt tuple_expr("struct", tuple_type);
      tuple_expr.copy_to_operands(
        from_double(q, result_type), from_double(r, result_type));
      return tuple_expr;
    }

    dividend = promote_to_double_if_needed(std::move(dividend));
    divisor = promote_to_double_if_needed(std::move(divisor));
  }
  else
  {
    exprt resolved_dividend = resolve_symbol(dividend);
    exprt resolved_divisor = resolve_symbol(divisor);
    if (
      resolved_dividend.is_constant() && resolved_divisor.is_constant() &&
      (resolved_divisor.type().is_signedbv() ||
       resolved_divisor.type().is_unsignedbv()))
    {
      const BigInt lhs_val = binary2integer(
        resolved_dividend.value().as_string(),
        resolved_dividend.type().is_signedbv());
      const BigInt rhs_val = binary2integer(
        resolved_divisor.value().as_string(),
        resolved_divisor.type().is_signedbv());
      if (rhs_val != 0)
      {
        BigInt q = lhs_val / rhs_val;
        BigInt r = lhs_val % rhs_val;
        if (r != 0 && ((lhs_val < 0) != (rhs_val < 0)))
        {
          q -= 1;
          r += rhs_val;
        }

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

        exprt tuple_expr("struct", tuple_type);
        tuple_expr.copy_to_operands(
          from_integer(q, result_type), from_integer(r, result_type));
        return tuple_expr;
      }
    }
  }

  // Calculate quotient: a // b (floor division)
  exprt quotient;
  if (result_type.is_floatbv())
  {
    // For floats, use floor(a / b)
    exprt div_expr("ieee_div", result_type);
    div_expr.copy_to_operands(dividend, divisor);

    side_effect_expr_function_callt floor_call;
    floor_call.function() =
      symbol_expr(get_c_math_symbol_cached("c:@F@floor", "floor"));
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
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value())
    return from_double(std::sin(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@sin", "sin", std::move(operand), element);
}

exprt python_math::handle_cos(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value())
    return from_double(std::cos(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@cos", "cos", std::move(operand), element);
}

exprt python_math::handle_exp(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value())
    return from_double(std::exp(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@exp", "exp", std::move(operand), element);
}

exprt python_math::handle_log(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value() && *val > 0.0)
    return from_double(std::log(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@log", "log", std::move(operand), element);
}

exprt python_math::handle_acos(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value() && *val >= -1.0 && *val <= 1.0)
    return from_double(std::acos(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@acos", "acos", std::move(operand), element);
}

exprt python_math::handle_atan(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value())
    return from_double(std::atan(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@atan", "atan", std::move(operand), element);
}

exprt python_math::handle_atan2(
  exprt y_operand,
  exprt x_operand,
  const nlohmann::json &element)
{
  if (std::optional<double> y_const = try_resolve_constant_double(y_operand),
      x_const = try_resolve_constant_double(x_operand);
      y_const.has_value() && x_const.has_value())
  {
    return from_double(std::atan2(*y_const, *x_const), double_type());
  }

  y_operand = promote_to_double_if_needed(std::move(y_operand));
  x_operand = promote_to_double_if_needed(std::move(x_operand));

  // Create the function call expression
  side_effect_expr_function_callt atan2_call;
  atan2_call.function() =
    symbol_expr(get_c_math_symbol_cached("c:@F@atan2", "atan2"));
  atan2_call.arguments() = {y_operand, x_operand};
  atan2_call.type() = double_type();
  atan2_call.location() = converter.get_location_from_decl(element);

  return atan2_call;
}

exprt python_math::handle_log2(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value() && *val > 0.0)
  {
    // Avoid platform/libm rounding noise for exact powers of two.
    int exponent = 0;
    const double mantissa = std::frexp(*val, &exponent);
    if (mantissa == 0.5)
      return from_double(static_cast<double>(exponent - 1), double_type());

    return from_double(std::log2(*val), double_type());
  }

  return build_unary_c_math_call(
    "c:@F@log2", "log2", std::move(operand), element);
}

exprt python_math::handle_pow(
  exprt base,
  exprt exp,
  const nlohmann::json &element)
{
  if (std::optional<double> base_const = try_resolve_constant_double(base),
      exp_const = try_resolve_constant_double(exp);
      base_const.has_value() && exp_const.has_value())
  {
    return from_double(std::pow(*base_const, *exp_const), double_type());
  }

  base = promote_to_double_if_needed(std::move(base));
  exp = promote_to_double_if_needed(std::move(exp));

  // Create the function call expression
  side_effect_expr_function_callt pow_call;
  pow_call.function() =
    symbol_expr(get_c_math_symbol_cached("c:@F@pow", "pow"));
  pow_call.arguments() = {base, exp};
  pow_call.type() = double_type();
  pow_call.location() = converter.get_location_from_decl(element);

  return pow_call;
}

exprt python_math::handle_fabs(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value())
    return from_double(std::fabs(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@fabs", "fabs", std::move(operand), element);
}

exprt python_math::handle_trunc(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value() && std::isfinite(*val))
  {
    const double truncated = std::trunc(*val);
    const double ll_min =
      static_cast<double>(std::numeric_limits<long long>::min());
    const double ll_max =
      static_cast<double>(std::numeric_limits<long long>::max());
    if (truncated >= ll_min && truncated <= ll_max)
      return from_integer(static_cast<long long>(truncated), int_type());
  }

  exprt trunc_call =
    build_unary_c_math_call("c:@F@trunc", "trunc", std::move(operand), element);
  exprt to_int("typecast", int_type());
  to_int.copy_to_operands(trunc_call);
  return to_int;
}

exprt python_math::handle_fmod(
  exprt lhs,
  exprt rhs,
  const nlohmann::json &element)
{
  if (std::optional<double> lhs_const = try_resolve_constant_double(lhs),
      rhs_const = try_resolve_constant_double(rhs);
      lhs_const.has_value() && rhs_const.has_value() && *rhs_const != 0.0)
  {
    return from_double(std::fmod(*lhs_const, *rhs_const), double_type());
  }

  lhs = promote_to_double_if_needed(std::move(lhs));
  rhs = promote_to_double_if_needed(std::move(rhs));

  // Create the function call expression
  side_effect_expr_function_callt fmod_call;
  fmod_call.function() =
    symbol_expr(get_c_math_symbol_cached("c:@F@fmod", "fmod"));
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
  if (std::optional<double> lhs_const = try_resolve_constant_double(lhs),
      rhs_const = try_resolve_constant_double(rhs);
      lhs_const.has_value() && rhs_const.has_value())
  {
    return from_double(std::copysign(*lhs_const, *rhs_const), double_type());
  }

  lhs = promote_to_double_if_needed(std::move(lhs));
  rhs = promote_to_double_if_needed(std::move(rhs));

  // Create the function call expression
  side_effect_expr_function_callt copysign_call;
  copysign_call.function() =
    symbol_expr(get_c_math_symbol_cached("c:@F@copysign", "copysign"));
  copysign_call.arguments() = {lhs, rhs};
  copysign_call.type() = double_type();
  copysign_call.location() = converter.get_location_from_decl(element);

  return copysign_call;
}

exprt python_math::handle_tan(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value())
    return from_double(std::tan(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@tan", "tan", std::move(operand), element);
}

exprt python_math::handle_asin(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value() && *val >= -1.0 && *val <= 1.0)
    return from_double(std::asin(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@asin", "asin", std::move(operand), element);
}

exprt python_math::handle_sinh(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value())
    return from_double(std::sinh(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@sinh", "sinh", std::move(operand), element);
}

exprt python_math::handle_cosh(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value())
    return from_double(std::cosh(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@cosh", "cosh", std::move(operand), element);
}

exprt python_math::handle_tanh(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value())
    return from_double(std::tanh(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@tanh", "tanh", std::move(operand), element);
}

exprt python_math::handle_log10(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value() && *val > 0.0)
    return from_double(std::log10(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@log10", "log10", std::move(operand), element);
}

exprt python_math::handle_expm1(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value())
    return from_double(std::expm1(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@expm1", "expm1", std::move(operand), element);
}

exprt python_math::handle_log1p(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value() && *val > -1.0)
    return from_double(std::log1p(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@log1p", "log1p", std::move(operand), element);
}

exprt python_math::handle_exp2(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value())
    return from_double(std::exp2(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@exp2", "exp2", std::move(operand), element);
}

exprt python_math::handle_asinh(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value())
    return from_double(std::asinh(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@asinh", "asinh", std::move(operand), element);
}

exprt python_math::handle_acosh(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value() && *val >= 1.0)
    return from_double(std::acosh(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@acosh", "acosh", std::move(operand), element);
}

exprt python_math::handle_atanh(exprt operand, const nlohmann::json &element)
{
  if (std::optional<double> val = try_resolve_constant_double(operand);
      val.has_value() && std::fabs(*val) < 1.0)
    return from_double(std::atanh(*val), double_type());

  return build_unary_c_math_call(
    "c:@F@atanh", "atanh", std::move(operand), element);
}

exprt python_math::handle_hypot(
  exprt lhs,
  exprt rhs,
  const nlohmann::json &element)
{
  if (std::optional<double> lhs_const = try_resolve_constant_double(lhs),
      rhs_const = try_resolve_constant_double(rhs);
      lhs_const.has_value() && rhs_const.has_value())
  {
    return from_double(std::hypot(*lhs_const, *rhs_const), double_type());
  }

  lhs = promote_to_double_if_needed(std::move(lhs));
  rhs = promote_to_double_if_needed(std::move(rhs));

  // Create the function call expression
  side_effect_expr_function_callt hypot_call;
  hypot_call.function() =
    symbol_expr(get_c_math_symbol_cached("c:@F@hypot", "hypot"));
  hypot_call.arguments() = {lhs, rhs};
  hypot_call.type() = double_type();
  hypot_call.location() = converter.get_location_from_decl(element);

  return hypot_call;
}

exprt python_math::handle_dist(exprt p, exprt q, const nlohmann::json &element)
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

  // Fast path: if both points resolve to concrete tuple literals, fold the
  // entire distance expression to a single constant.
  exprt resolved_p = resolve_symbol(p);
  exprt resolved_q = resolve_symbol(q);
  if (
    resolved_p.id() == "struct" && resolved_q.id() == "struct" &&
    resolved_p.operands().size() == p_size &&
    resolved_q.operands().size() == q_size)
  {
    double sum_sq = 0.0;
    bool all_constant_numeric = true;
    for (size_t i = 0; i < p_size; ++i)
    {
      std::optional<double> pi =
        try_resolve_constant_double(resolved_p.operands()[i]);
      std::optional<double> qi =
        try_resolve_constant_double(resolved_q.operands()[i]);
      if (!pi.has_value() || !qi.has_value())
      {
        all_constant_numeric = false;
        break;
      }
      const double diff = *pi - *qi;
      sum_sq += diff * diff;
    }

    if (all_constant_numeric)
      return from_double(std::sqrt(sum_sq), double_type());
  }

  // Build sum of squared differences: (p[0]-q[0])^2 + (p[1]-q[1])^2 + ...
  // Use ieee_* operations for floating-point arithmetic
  exprt rounding_mode = symbol_exprt("c:@__ESBMC_rounding_mode", int_type());

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
