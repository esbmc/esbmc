#include <python-frontend/math/bounded_math.h>
#include <python-frontend/python_converter.h>
#include <python-frontend/exception/python_exception_handler.h>
#include <irep2/irep2_utils.h>
#include <util/arith/arith_tools.h>
#include <util/arith/ieee_float.h>
#include <util/arith/transcendental.h>
#include <util/irep/migrate.h>
#include <util/lang/c_types.h>
#include <util/irep/std_code.h>
#include <util/irep/std_expr.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <numbers>
#include <optional>
#include <set>

namespace
{
const double inf = INFINITY;

exprt between(const exprt &value, double low, double high)
{
  return and_exprt(
    binary_relation_exprt(value, ">=", from_double(low, double_type())),
    binary_relation_exprt(value, "<=", from_double(high, double_type())));
}

// An exact value at a signed zero, where the bounds cannot keep the sign.
exprt at_zero(const std::string &function, const exprt &x)
{
  static const std::set<std::string> one = {
    "exp", "exp2", "cos", "cosh", "erfc"};
  if (one.count(function) != 0)
    return from_double(1.0, double_type());
  if (transcendental::keeps_signed_zero(function))
    return x;
  return nil_exprt();
}

exprt is_nan(const exprt &x)
{
  expr2tc x2;
  migrate_expr(x, x2);
  return migrate_expr_back(isnan2tc(x2));
}

exprt is_infinite(const exprt &x)
{
  expr2tc x2;
  migrate_expr(x, x2);
  return migrate_expr_back(isinf2tc(x2));
}

exprt abs_value(const exprt &x)
{
  expr2tc x2;
  migrate_expr(x, x2);
  return migrate_expr_back(abs2tc(x2->type, x2));
}

exprt sign_bit(const exprt &x)
{
  expr2tc x2;
  migrate_expr(x, x2);
  return migrate_expr_back(typecast2tc(get_bool_type(), signbit2tc(x2)));
}

exprt number(double value)
{
  return from_double(value, double_type());
}

// x rounded to an integral value in the IEEE rounding mode `mode`.
exprt rounded(const exprt &x, ieee_floatt::rounding_modet mode)
{
  expr2tc x2;
  migrate_expr(x, x2);
  return migrate_expr_back(nearbyint2tc(
    x2->type, x2, constant_int2tc(get_int32_type(), BigInt(mode))));
}

exprt is_integral(const exprt &x)
{
  return and_exprt(
    not_exprt(or_exprt(is_nan(x), is_infinite(x))),
    equality_exprt(rounded(x, ieee_floatt::ROUND_TO_ZERO), x));
}

// floor(x) is even; x finite, below 2^53 in magnitude.
exprt even_floor(const exprt &x)
{
  const exprt floor = rounded(x, ieee_floatt::ROUND_TO_MINUS_INF);
  exprt half("ieee_div", double_type());
  half.copy_to_operands(floor, number(2.0));
  half.add("rounding_mode") =
    symbol_exprt("c:@__ESBMC_rounding_mode", int_type());
  return is_integral(half);
}

// c with the sign of y, as C's copysign; y may be a signed zero.
exprt with_sign_of(double c, const exprt &y)
{
  return if_exprt(sign_bit(y), number(-c), number(c));
}

// A constant of arctan2 widened by a relative 2^-48 on each side, more than
// any implementation's rounding.
double below(double c)
{
  return c * (1 - 0x1p-48);
}

double above(double c)
{
  return c * (1 + 0x1p-48);
}

exprt arithmetic(const irep_idt &op, const exprt &a, const exprt &b)
{
  exprt result(op, double_type());
  result.copy_to_operands(a, b);
  return result;
}

exprt between(const exprt &value, const exprt &low, const exprt &high)
{
  return and_exprt(
    binary_relation_exprt(value, ">=", low),
    binary_relation_exprt(value, "<=", high));
}

// Where math.* raises "math domain error": outside the domain, at a pole, or
// at an infinity of a periodic function.
exprt domain_error(const std::string &function, const exprt &x)
{
  const transcendental::domain domain = transcendental::domain_of(function);
  exprt::operandst outside = {not_exprt(between(x, domain.low, domain.high))};
  for (double pole : domain.poles)
    outside.push_back(equality_exprt(x, from_double(pole, double_type())));
  if (domain.periodic)
    outside.push_back(is_infinite(x));
  return and_exprt(not_exprt(is_nan(x)), or_exprt(outside));
}

// The powers every implementation computes exactly: an integral result of
// an integral base below 2^53, a power of two that stays normal, and the
// special values C99 F.9.4.4 fixes.
bool exact_power(double x, double y, double value)
{
  if (!std::isfinite(value) || value == 0.0 || y == 0.0 || x == 1.0 || y == 1.0)
    return true;
  if (!std::isfinite(x) || !std::isfinite(y) || y != std::floor(y))
    return false;
  int exponent = 0;
  const bool power_of_two = std::frexp(std::fabs(x), &exponent) == 0.5;
  if (power_of_two)
    return std::fabs(value) >= DBL_MIN;
  return x == std::floor(x) && y > 0 && std::fabs(value) <= 0x1p53;
}

// The value of a number literal, int or float, possibly cast to double.
std::optional<double> literal_double(const exprt &e)
{
  if (e.id() == "typecast")
    return literal_double(e.op0());
  if (e.id() == "unary-")
  {
    const std::optional<double> operand = literal_double(e.op0());
    return operand ? std::optional<double>(-*operand) : std::nullopt;
  }
  const exprt &x = e;
  if (!x.is_constant())
    return std::nullopt;
  if (x.type().is_floatbv())
    return ieee_floatt(to_constant_expr(x)).to_double();
  BigInt value;
  if (
    (x.type().is_signedbv() || x.type().is_unsignedbv()) &&
    !to_integer(x, value))
    return value.to_int64();
  return std::nullopt;
}

exprt odd_integer(const exprt &y)
{
  exprt half("ieee_div", double_type());
  half.copy_to_operands(y, number(2.0));
  half.add("rounding_mode") =
    symbol_exprt("c:@__ESBMC_rounding_mode", int_type());
  return and_exprt(is_integral(y), not_exprt(is_integral(half)));
}

exprt ieee(const irep_idt &op, const exprt &a, const exprt &b)
{
  exprt result(op, double_type());
  result.copy_to_operands(a, b);
  result.add("rounding_mode") =
    symbol_exprt("c:@__ESBMC_rounding_mode", int_type());
  return result;
}

// Whether a * b rounds to `product` exactly, by fma(a, b, -product) == 0.
exprt exact_product(const exprt &a, const exprt &b, const exprt &product)
{
  expr2tc a2, b2, p2;
  migrate_expr(a, a2);
  migrate_expr(b, b2);
  migrate_expr(product, p2);
  const expr2tc rm = symbol2tc(get_int32_type(), "c:@__ESBMC_rounding_mode");
  const expr2tc residual =
    ieee_fma2tc(a2->type, a2, b2, neg2tc(p2->type, p2), rm);
  return migrate_expr_back(equality2tc(residual, gen_zero(residual->type)));
}

exprt negated(const exprt &x)
{
  exprt result("unary-", double_type());
  result.copy_to_operands(x);
  return result;
}

exprt less(const exprt &a, double b)
{
  return binary_relation_exprt(a, "<", number(b));
}

exprt greater(const exprt &a, double b)
{
  return binary_relation_exprt(a, ">", number(b));
}

exprt finite(const exprt &x)
{
  return not_exprt(or_exprt(is_nan(x), is_infinite(x)));
}

// The exponent e of a >= 0 written m * 2^e with m in [0.5, 1), from its bits;
// `low` asks for the bound below a subnormal's, `high` for the one above.
exprt binary_exponent(const exprt &a, bool high)
{
  expr2tc a2;
  migrate_expr(a, a2);
  const expr2tc bits = bitcast2tc(get_uint64_type(), a2);
  const expr2tc biased = typecast2tc(
    get_int64_type(),
    lshr2tc(get_uint64_type(), bits, constant_int2tc(get_uint64_type(), 52)));
  const expr2tc normal = sub2tc(
    get_int64_type(),
    biased,
    constant_int2tc(get_int64_type(), high ? 1022 : 1023));
  const expr2tc subnormal =
    constant_int2tc(get_int64_type(), high ? BigInt(-1021) : BigInt(-1075));
  const expr2tc exponent = if2tc(
    get_int64_type(),
    equality2tc(biased, constant_int2tc(get_int64_type(), 0)),
    subnormal,
    normal);
  return migrate_expr_back(typecast2tc(migrate_type(double_type()), exponent));
}
} // namespace

bounded_math::bounded_math(
  python_converter &converter,
  const nlohmann::json &call)
  : converter_(converter), call_(call)
{
}

bool bounded_math::is_bounded(const std::string &function)
{
  return transcendental::is_bounded(function);
}

bool bounded_math::is_bounded_binary(const std::string &function)
{
  return transcendental::is_bounded_binary(function);
}

bool bounded_math::has_runtime_model(const std::string &function)
{
  return (is_bounded(function) &&
          !transcendental::cpython_implements(function)) ||
         function == "sqrt";
}

double bounded_math::host_value(const std::string &function, double x)
{
  return transcendental::host_value(function, x);
}

bool bounded_math::outside_domain(const std::string &function, double x)
{
  const transcendental::domain domain = transcendental::domain_of(function);
  return !(x >= domain.low && x <= domain.high) ||
         std::find(domain.poles.begin(), domain.poles.end(), x) !=
           domain.poles.end();
}

exprt bounded_math::literal(const std::string &function, double x)
{
  const transcendental::interval result = transcendental::at(function, x);
  if (result.low == result.high || std::isnan(result.low))
    return from_double(result.low, double_type());
  return within(result.low, result.high);
}

exprt bounded_math::runtime(
  const std::string &function,
  const exprt &x,
  math_module module)
{
  const exprt argument = read_once(x);
  if (function == "sqrt")
    return converter_.get_math_handler().handle_sqrt(argument, call_);
  if (transcendental::cpython_implements(function))
    return gamma_family(function, argument);
  const exprt value =
    special_cases(function, argument, bounded_call(function, argument, module));
  if (module == math_module::math)
    raise_math_errors(function, argument, value);
  return value;
}

exprt bounded_math::binary(
  const std::string &function,
  const exprt &a,
  const exprt &b,
  math_module module)
{
  if (function == "pow")
    return power(a, b, module);
  const exprt x = read_once(a), y = read_once(b);
  if (x.is_constant() && y.is_constant())
  {
    const double first = ieee_floatt(to_constant_expr(x)).to_double();
    const double second = ieee_floatt(to_constant_expr(y)).to_double();
    const double value = transcendental::host_value(function, first, second);
    // A zero, infinite or NaN argument gives a value C99 F.9 fixes exactly.
    const bool special = first == 0.0 || second == 0.0 ||
                         !std::isfinite(first) || !std::isfinite(second);
    const transcendental::interval result =
      module == math_module::math || special
        ? transcendental::interval{value, value}
        : transcendental::within_ulps(value);
    return result.low == result.high || std::isnan(result.low)
             ? number(result.low)
             : within(result.low, result.high);
  }
  return function == "hypot" ? hypot(x, y) : arctan2(x, y);
}

// pow(x, y) of math.pow (raising Python's errors) or numpy.power: the
// special values of C99 F.9.4.4 are exact, and otherwise the result has the
// sign an odd integral exponent gives a negative base, and the magnitude
// power_magnitude bounds.
exprt bounded_math::power(const exprt &a, const exprt &b, math_module module)
{
  const exprt x = read_once(a), y = read_once(b);
  const std::optional<double> base = literal_double(x),
                              exponent = literal_double(y);
  if (base && exponent)
    return constant_power(*base, *exponent, module);
  const exprt zero = number(0.0);
  const exprt odd = odd_integer(y);
  const exprt domain = and_exprt(
    and_exprt(less(x, 0.0), finite(x)),
    and_exprt(finite(y), not_exprt(is_integral(y))));
  const exprt regular = and_exprt(
    and_exprt(finite(x), finite(y)),
    and_exprt(
      and_exprt(
        not_exprt(equality_exprt(x, zero)), not_exprt(equality_exprt(y, zero))),
      and_exprt(not_exprt(equality_exprt(x, number(1.0))), not_exprt(domain))));
  const exprt magnitude = power_magnitude(abs_value(x), y, regular);
  const exprt general =
    if_exprt(and_exprt(less(x, 0.0), odd), negated(magnitude), magnitude);
  const exprt result = if_exprt(
    or_exprt(equality_exprt(y, zero), equality_exprt(x, number(1.0))),
    number(1.0),
    if_exprt(
      or_exprt(is_nan(x), is_nan(y)),
      number(NAN),
      if_exprt(
        is_infinite(y),
        power_infinite_exponent(x, y),
        if_exprt(
          is_infinite(x),
          power_infinite_base(x, y, odd),
          if_exprt(
            equality_exprt(x, zero),
            power_zero_base(x, y, odd),
            if_exprt(domain, number(NAN), general))))));
  if (module == math_module::math)
  {
    raise_if(
      or_exprt(
        domain,
        and_exprt(equality_exprt(x, zero), and_exprt(less(y, 0.0), finite(y)))),
      "ValueError",
      "math domain error");
    raise_if(
      and_exprt(and_exprt(finite(x), finite(y)), is_infinite(result)),
      "OverflowError",
      "math range error");
  }
  return result;
}

exprt bounded_math::power_infinite_exponent(const exprt &x, const exprt &y)
{
  const exprt ax = abs_value(x);
  return if_exprt(
    equality_exprt(ax, number(1.0)),
    number(1.0),
    if_exprt(
      equality_exprt(less(ax, 1.0), greater(y, 0.0)),
      number(0.0),
      number(inf)));
}

exprt bounded_math::power_infinite_base(
  const exprt &x,
  const exprt &y,
  const exprt &odd)
{
  const exprt positive_y = greater(y, 0.0);
  return if_exprt(
    greater(x, 0.0),
    if_exprt(positive_y, number(inf), number(0.0)),
    if_exprt(
      positive_y,
      if_exprt(odd, number(-inf), number(inf)),
      if_exprt(odd, number(-0.0), number(0.0))));
}

exprt bounded_math::power_zero_base(
  const exprt &x,
  const exprt &y,
  const exprt &odd)
{
  return if_exprt(
    greater(y, 0.0),
    if_exprt(odd, x, number(0.0)),
    if_exprt(odd, with_sign_of(inf, x), number(inf)));
}

// |x|^y for x and y finite and away from the special values: within the
// ulps of the product for a literal integral exponent up to 8 in magnitude
// (sqrt for 0.5), on the side of 1 and of |x| monotonicity gives, and
// infinite exactly where the binary exponents of |x| and y say it overflows.
// A factor of 2^-48 covers every implementation's rounding.
exprt bounded_math::power_magnitude(
  const exprt &a,
  const exprt &y,
  const exprt &regular)
{
  const exprt m = fresh("$math$", nil_exprt());
  const exprt one = number(1.0);
  const exprt down = number(1 - 0x1p-48), up = number(1 + 0x1p-48);
  const exprt above_one = or_exprt(
    and_exprt(greater(a, 1.0), greater(y, 0.0)),
    and_exprt(less(a, 1.0), less(y, 0.0)));
  const exprt below_one = or_exprt(
    and_exprt(greater(a, 1.0), less(y, 0.0)),
    and_exprt(less(a, 1.0), greater(y, 0.0)));
  const exprt a_down = ieee("ieee_mul", a, down),
              a_up = ieee("ieee_mul", a, up);
  const exprt beyond_a = if_exprt(
    binary_relation_exprt(a, ">=", one),
    binary_relation_exprt(m, ">=", a_down),
    binary_relation_exprt(m, "<=", a_up));
  const exprt within_a = if_exprt(
    binary_relation_exprt(a, ">=", one),
    binary_relation_exprt(m, "<=", a_up),
    binary_relation_exprt(m, ">=", a_down));
  const exprt shape = and_exprt(
    and_exprt(
      binary_relation_exprt(m, ">=", number(0.0)),
      and_exprt(
        or_exprt(not_exprt(above_one), binary_relation_exprt(m, ">=", down)),
        or_exprt(not_exprt(below_one), binary_relation_exprt(m, "<=", up)))),
    and_exprt(
      or_exprt(binary_relation_exprt(y, "<", one), beyond_a),
      or_exprt(or_exprt(less(y, 0.0), greater(y, 1.0)), within_a)));
  assume(or_exprt(
    not_exprt(regular),
    and_exprt(
      shape,
      and_exprt(power_overflow(a, y, m), power_literal_exponent(a, y, m)))));
  return m;
}

// a^y overflows for a >= 1 < y once y * e_low >= 1025, and cannot while
// y * e_high <= 1022, where 2^(e_low - 1) <= a < 2^e_high; likewise for
// a < 1 and y < 0 with (1 / a)^-y.
exprt bounded_math::power_overflow(
  const exprt &a,
  const exprt &y,
  const exprt &m)
{
  const exprt high = binary_exponent(a, true), low = binary_exponent(a, false);
  const exprt neg_y = negated(y);
  const exprt grows =
    and_exprt(binary_relation_exprt(a, ">=", number(1.0)), greater(y, 0.0));
  const exprt shrinks = and_exprt(less(a, 1.0), less(y, 0.0));
  const exprt safe = or_exprt(
    and_exprt(
      grows,
      binary_relation_exprt(ieee("ieee_mul", y, high), "<=", number(1022.0))),
    and_exprt(
      shrinks,
      binary_relation_exprt(
        ieee("ieee_mul", neg_y, negated(low)), "<=", number(1022.0))));
  const exprt sure = or_exprt(
    and_exprt(
      grows,
      binary_relation_exprt(
        ieee("ieee_mul", y, ieee("ieee_sub", low, number(1.0))),
        ">=",
        number(1025.0))),
    and_exprt(
      shrinks,
      binary_relation_exprt(
        ieee("ieee_mul", neg_y, negated(high)), ">=", number(1025.0))));
  return and_exprt(
    or_exprt(not_exprt(safe), finite(m)),
    or_exprt(not_exprt(sure), is_infinite(m)));
}

// A literal exponent of 1 keeps |x|; 0.5 is sqrt and 2..8 in magnitude the
// product (its reciprocal for a negative one), within the ulps of the
// library plus the product's own rounding while the product stays normal.
exprt bounded_math::power_literal_exponent(
  const exprt &a,
  const exprt &y,
  const exprt &m)
{
  const std::optional<double> literal = literal_double(y);
  if (!literal)
    return true_exprt();
  const double exponent = *literal;
  if (exponent == 1.0)
    return equality_exprt(m, a);
  exprt product;
  std::size_t steps = 0;
  if (exponent == 0.5)
  {
    exprt root("ieee_sqrt", double_type());
    root.copy_to_operands(a);
    root.add("rounding_mode") =
      symbol_exprt("c:@__ESBMC_rounding_mode", int_type());
    product = root;
    steps = 1;
  }
  else if (
    exponent == std::floor(exponent) && std::fabs(exponent) >= 2 &&
    std::fabs(exponent) <= 8)
  {
    steps = static_cast<std::size_t>(std::fabs(exponent));
    product = a;
    exprt::operandst exact;
    for (std::size_t k = 1; k < steps; ++k)
    {
      const exprt next = ieee("ieee_mul", product, a);
      exact.push_back(exact_product(product, a, next));
      product = next;
    }
    // Every implementation returns a product it can represent exactly.
    if (exponent > 0)
      return or_exprt(
        and_exprt(and_exprt(exact), equality_exprt(m, product)),
        and_exprt(not_exprt(and_exprt(exact)), relative_to(product, m, steps)));
    product = ieee("ieee_div", number(1.0), product);
  }
  else
    return true_exprt();
  return relative_to(product, m, steps);
}

// m within the library's ulps of `product`, which rounded `steps` times,
// while the product stays normal.
exprt bounded_math::relative_to(
  const exprt &product,
  const exprt &m,
  std::size_t steps)
{
  const double tolerance =
    (2.0 * steps + 2.0 * transcendental::max_ulps() + 4.0) * 0x1p-53;
  const exprt normal = and_exprt(
    binary_relation_exprt(product, ">=", number(0x1p-1000)),
    binary_relation_exprt(product, "<=", number(0x1p1000)));
  const exprt radius = ieee("ieee_mul", product, number(tolerance));
  return or_exprt(
    not_exprt(normal),
    between(
      m, ieee("ieee_sub", product, radius), ieee("ieee_add", product, radius)));
}

// pow of two literals: math.* the C library's value and its errors, raised
// where the call runs; numpy.* within the ulps unless every implementation
// is exact.
exprt bounded_math::constant_power(
  double base,
  double exponent,
  math_module module)
{
  const double value = std::pow(base, exponent);
  const bool finite_args = std::isfinite(base) && std::isfinite(exponent);
  if (module == math_module::math)
  {
    const bool domain =
      finite_args && ((base < 0 && exponent != std::floor(exponent)) ||
                      (base == 0 && exponent < 0));
    if (domain)
      raise_if(true_exprt(), "ValueError", "math domain error");
    else if (finite_args && std::isinf(value))
      raise_if(true_exprt(), "OverflowError", "math range error");
    return number(domain ? NAN : value);
  }
  if (exact_power(base, exponent, value))
    return number(value);
  const transcendental::interval bound = transcendental::within_ulps(value);
  return within(bound.low, bound.high);
}

// An infinity wins over NaN, a zero argument leaves the other's magnitude,
// and otherwise the result lies between the larger magnitude and the sum.
exprt bounded_math::hypot(const exprt &x, const exprt &y)
{
  const exprt ax = abs_value(x), ay = abs_value(y);
  const exprt zero = number(0.0);
  const exprt regular = and_exprt(
    and_exprt(
      not_exprt(or_exprt(is_infinite(x), is_infinite(y))),
      not_exprt(or_exprt(is_nan(x), is_nan(y)))),
    and_exprt(
      not_exprt(equality_exprt(x, zero)), not_exprt(equality_exprt(y, zero))));
  const exprt larger = if_exprt(binary_relation_exprt(ax, ">=", ay), ax, ay);
  // hypot <= min(|x| + |y|, sqrt(2) * max(|x|, |y|)); the factor allows any
  // rounding of an implementation, and the bound an overflow to infinity.
  const exprt sum = arithmetic("+", ax, ay);
  const exprt diagonal = arithmetic("*", larger, number(std::numbers::sqrt2));
  const exprt upper = arithmetic(
    "*",
    if_exprt(binary_relation_exprt(sum, "<=", diagonal), sum, diagonal),
    number(1 + 0x1p-48));
  // Constrained only for regular arguments, so the paths of NaN and infinite
  // arguments survive the assumption.
  const exprt result = fresh("$math$", nil_exprt());
  assume(or_exprt(
    not_exprt(regular),
    and_exprt(between(result, larger, upper), squares_relation(x, y, result))));
  return if_exprt(
    or_exprt(is_infinite(x), is_infinite(y)),
    number(inf),
    if_exprt(
      or_exprt(is_nan(x), is_nan(y)),
      number(NAN),
      if_exprt(
        equality_exprt(x, zero),
        ay,
        if_exprt(equality_exprt(y, zero), ax, result))));
}

// r^2 within the ulps of x^2 + y^2, while that sum stays normal: r is within
// max_ulps of the exact hypot, and each square and the sum round once more.
exprt bounded_math::squares_relation(
  const exprt &x,
  const exprt &y,
  const exprt &r)
{
  const exprt sum =
    ieee("ieee_add", ieee("ieee_mul", x, x), ieee("ieee_mul", y, y));
  const exprt normal = and_exprt(greater(sum, 0x1p-1000), less(sum, 0x1p1000));
  const exprt radius = ieee(
    "ieee_mul",
    sum,
    number((2 * transcendental::max_ulps() + 16) * DBL_EPSILON));
  const exprt square = ieee("ieee_mul", r, r);
  return or_exprt(
    not_exprt(normal),
    between(
      square, ieee("ieee_sub", sum, radius), ieee("ieee_add", sum, radius)));
}

// The special values of C99 F.9.1.4 are exact; otherwise the result lies in
// the quadrant of (x, y), on the side of pi/4 or 3*pi/4 that |y| >= |x| picks,
// and is the magnitude of what symex gives arguments that are constant by
// then: the C library's value within the ulps bound.
exprt bounded_math::arctan2(const exprt &y, const exprt &x)
{
  static const double pi = std::numbers::pi, half_pi = pi / 2,
                      quarter_pi = pi / 4, three_quarter_pi = 3 * pi / 4;
  const exprt zero = number(0.0);
  const exprt x_positive = binary_relation_exprt(x, ">", zero);
  const exprt regular = and_exprt(
    and_exprt(
      not_exprt(or_exprt(is_infinite(x), is_infinite(y))),
      not_exprt(or_exprt(is_nan(x), is_nan(y)))),
    and_exprt(
      not_exprt(equality_exprt(x, zero)), not_exprt(equality_exprt(y, zero))));
  const exprt steep = binary_relation_exprt(abs_value(y), ">=", abs_value(x));
  const exprt magnitude = fresh("$math$", nil_exprt());
  const exprt in_quadrant = if_exprt(
    x_positive,
    if_exprt(
      steep,
      between(magnitude, number(below(quarter_pi)), number(above(half_pi))),
      between(magnitude, zero, number(above(quarter_pi)))),
    if_exprt(
      steep,
      between(
        magnitude, number(below(half_pi)), number(above(three_quarter_pi))),
      between(magnitude, number(below(three_quarter_pi)), number(above(pi)))));
  const exprt library = bounded_binary_call("arctan2", y, x);
  assume(or_exprt(
    not_exprt(regular),
    and_exprt(in_quadrant, equality_exprt(magnitude, abs_value(library)))));
  exprt negated("unary-", double_type());
  negated.copy_to_operands(magnitude);
  const exprt general = if_exprt(sign_bit(y), negated, magnitude);
  const exprt at_infinite_y = if_exprt(
    is_infinite(x),
    if_exprt(
      x_positive,
      with_sign_of(quarter_pi, y),
      with_sign_of(three_quarter_pi, y)),
    with_sign_of(half_pi, y));
  const exprt at_infinite_x =
    if_exprt(x_positive, with_sign_of(0.0, y), with_sign_of(pi, y));
  const exprt at_zero_y = if_exprt(
    and_exprt(equality_exprt(x, zero), not_exprt(sign_bit(x))),
    y,
    if_exprt(x_positive, y, with_sign_of(pi, y)));
  return if_exprt(
    or_exprt(is_nan(x), is_nan(y)),
    number(NAN),
    if_exprt(
      equality_exprt(y, zero),
      at_zero_y,
      if_exprt(
        is_infinite(y),
        at_infinite_y,
        if_exprt(
          is_infinite(x),
          at_infinite_x,
          if_exprt(
            equality_exprt(x, zero), with_sign_of(half_pi, y), general)))));
}

// Symex gives a constant argument the module's literal result (the C
// library's exact value for math.*) and otherwise the bounds of its piece.
exprt bounded_math::bounded_call(
  const std::string &function,
  const exprt &x,
  math_module module)
{
  const std::string name = "__builtin_bounded_" + function;
  const std::string id = "c:@F@" + name;
  code_typet type;
  type.return_type() = double_type();
  type.arguments().push_back(code_typet::argumentt(double_type()));
  type.arguments().push_back(code_typet::argumentt(bool_type()));
  if (!converter_.symbol_table_.find_symbol(id))
    converter_.add_symbol(converter_.create_symbol(
      converter_.python_file(),
      name,
      id,
      converter_.get_location_from_decl(call_),
      type));
  code_function_callt call;
  call.function() = symbol_exprt(id, type);
  call.arguments().push_back(x);
  call.arguments().push_back(
    module == math_module::math ? exprt(true_exprt()) : false_exprt());
  call.type() = double_type();
  call.location() = converter_.get_location_from_decl(call_);
  return fresh("$math$", call);
}

// Symex gives arguments that are constant by then the C library's value
// within the ulps bound (exact at C99's special values), and anything else
// no constraint.
exprt bounded_math::bounded_binary_call(
  const std::string &function,
  const exprt &a,
  const exprt &b)
{
  const std::string name = "__builtin_bounded_" + function;
  const std::string id = "c:@F@" + name;
  code_typet type;
  type.return_type() = double_type();
  type.arguments().push_back(code_typet::argumentt(double_type()));
  type.arguments().push_back(code_typet::argumentt(double_type()));
  if (!converter_.symbol_table_.find_symbol(id))
    converter_.add_symbol(converter_.create_symbol(
      converter_.python_file(),
      name,
      id,
      converter_.get_location_from_decl(call_),
      type));
  code_function_callt call;
  call.function() = symbol_exprt(id, type);
  call.arguments().push_back(a);
  call.arguments().push_back(b);
  call.type() = double_type();
  call.location() = converter_.get_location_from_decl(call_);
  return fresh("$math$", call);
}

// The value of a number literal, int or float.
static std::optional<double> constant_value(const exprt &x)
{
  if (!x.is_constant())
    return std::nullopt;
  if (x.type().is_floatbv())
    return ieee_floatt(to_constant_expr(x)).to_double();
  if (x.type().is_signedbv() || x.type().is_unsignedbv())
  {
    BigInt value;
    if (!to_integer(x, value))
      return value.to_int64();
  }
  return std::nullopt;
}

// An infinite argument wins over NaN, and NaN over a sum that overflows, as
// CPython checks the arguments before it sums their squares.
exprt bounded_math::hypot_of(const std::vector<exprt> &args)
{
  bool constant = true, infinite = false, nan = false;
  double value = 0.0;
  for (const exprt &x : args)
  {
    const std::optional<double> literal = constant_value(x);
    constant = constant && literal.has_value();
    if (!constant)
      break;
    infinite = infinite || std::isinf(*literal);
    nan = nan || std::isnan(*literal);
    value = transcendental::host_value("hypot", value, *literal);
  }
  if (!constant)
    return hypot_of_runtime(args);
  if (infinite || nan)
    return number(infinite ? inf : NAN);
  // CPython rounds a sum of more than two squares once, where each step here
  // rounds again.
  if (args.size() <= 2 || value == 0.0 || !std::isfinite(value))
    return from_double(value, double_type());
  const transcendental::interval bound = transcendental::within_ulps(value);
  return within(bound.low, bound.high);
}

exprt bounded_math::hypot_of_runtime(const std::vector<exprt> &args)
{
  std::vector<exprt> operands;
  exprt::operandst infinite, nan;
  for (const exprt &x : args)
  {
    operands.push_back(read_once(x));
    infinite.push_back(is_infinite(operands.back()));
    nan.push_back(is_nan(operands.back()));
  }
  operands.resize(std::max<std::size_t>(operands.size(), 2), number(0.0));
  exprt result = binary("hypot", operands[0], operands[1], math_module::math);
  for (std::size_t i = 2; i < operands.size(); ++i)
    result = binary("hypot", result, operands[i], math_module::math);
  if (args.size() <= 2)
    return result;
  return if_exprt(
    or_exprt(infinite),
    number(inf),
    if_exprt(or_exprt(nan), number(NAN), result));
}

bool bounded_math::is_float_predicate(const std::string &function)
{
  return function == "isnan" || function == "isinf" || function == "isfinite";
}

exprt bounded_math::float_predicate(const std::string &function, const exprt &x)
{
  const exprt value = read_once(x);
  if (function == "isnan")
    return is_nan(value);
  if (function == "isinf")
    return is_infinite(value);
  return not_exprt(or_exprt(is_nan(value), is_infinite(value)));
}

// NaN propagates, an argument outside the domain gives NaN, and a signed
// zero, a pole or an infinity has an exact result.
exprt bounded_math::special_cases(
  const std::string &function,
  const exprt &x,
  exprt result)
{
  const transcendental::domain domain = transcendental::domain_of(function);
  const exprt zero = at_zero(function, x);
  if (!zero.is_nil())
    result = if_exprt(
      equality_exprt(x, from_double(0.0, double_type())), zero, result);
  for (double pole : domain.poles)
    result = if_exprt(
      equality_exprt(x, from_double(pole, double_type())),
      from_double(host_value(function, pole), double_type()),
      result);
  const exprt positive =
    binary_relation_exprt(x, ">", from_double(0.0, double_type()));
  result = if_exprt(
    is_infinite(x),
    if_exprt(positive, literal(function, inf), literal(function, -inf)),
    result);
  result = if_exprt(
    not_exprt(between(x, domain.low, domain.high)),
    from_double(NAN, double_type()),
    result);
  return if_exprt(is_nan(x), x, result);
}

// math.gamma or math.lgamma: the pieces bound a positive argument; a zero, a
// negative integer and gamma's -inf are poles ("math domain error"), lgamma's
// -inf is +inf, and between the negative poles gamma has the sign of
// (-1)^ceil(-x) and lgamma is finite. Only math.* has them; NumPy does not.
exprt bounded_math::gamma_family(const std::string &function, const exprt &x)
{
  const bool gamma = function == "gamma";
  const exprt zero = number(0.0);
  // The pieces cover x > 0 only, so a constant x <= 0 must not reach them.
  const exprt positive_part = bounded_call(
    function,
    if_exprt(binary_relation_exprt(x, ">", zero), x, number(1.0)),
    math_module::math);
  // CPython's gamma of a tiny x is 1/x, which overflows below this magnitude.
  const exprt tiny_negative = and_exprt(
    binary_relation_exprt(x, "<", zero),
    binary_relation_exprt(x, ">", number(-5.5626846462680084e-309)));
  const exprt pole =
    and_exprt(binary_relation_exprt(x, "<=", zero), is_integral(x));
  const exprt negative_infinity =
    and_exprt(is_infinite(x), binary_relation_exprt(x, "<", zero));
  const exprt between_poles = fresh("$math$", nil_exprt());
  const exprt finite =
    not_exprt(or_exprt(is_nan(between_poles), is_infinite(between_poles)));
  // gamma only underflows to zero well below x = -170.
  const exprt underflow = and_exprt(
    binary_relation_exprt(x, "<", number(-170.0)),
    equality_exprt(between_poles, zero));
  assume(
    gamma ? and_exprt(
              finite,
              or_exprt(
                underflow,
                if_exprt(
                  even_floor(x),
                  binary_relation_exprt(between_poles, ">", zero),
                  binary_relation_exprt(between_poles, "<", zero))))
          : finite);
  const exprt result = if_exprt(
    is_nan(x),
    x,
    if_exprt(
      negative_infinity,
      number(gamma ? NAN : inf),
      if_exprt(
        pole,
        number(gamma ? NAN : inf),
        if_exprt(
          binary_relation_exprt(x, "<", zero),
          gamma ? if_exprt(tiny_negative, number(-inf), between_poles)
                : between_poles,
          positive_part))));
  raise_if(
    or_exprt(pole, gamma ? negative_infinity : false_exprt()),
    "ValueError",
    "math domain error");
  raise_if(
    and_exprt(
      not_exprt(or_exprt(is_nan(x), is_infinite(x))), is_infinite(result)),
    "OverflowError",
    "math range error");
  return result;
}

exprt bounded_math::fresh(const std::string &name, const exprt &value)
{
  const typet type = value.is_nil() ? double_type() : value.type();
  symbolt &tmp = converter_.create_tmp_symbol(call_, name, type, exprt());
  const exprt symbol = symbol_expr(tmp);
  const locationt location = converter_.get_location_from_decl(call_);
  code_declt decl(symbol);
  decl.location() = location;
  converter_.add_instruction(decl);
  if (value.id() == "code" && value.get("statement") == "function_call")
  {
    code_function_callt call = to_code_function_call(to_code(value));
    call.lhs() = symbol;
    converter_.add_instruction(call);
    return symbol;
  }
  code_assignt assign(
    symbol, value.is_nil() ? side_effect_expr_nondett(type) : value);
  assign.location() = location;
  converter_.add_instruction(assign);
  return symbol;
}

// A number literal, such as -52.0, reads nothing, so it stays a literal.
exprt bounded_math::capture(const exprt &x)
{
  return x.is_constant() || literal_double(x) ? x : fresh("$math_arg$", x);
}

exprt bounded_math::read_once(const exprt &x)
{
  if (const std::optional<double> literal = literal_double(x))
    return number(*literal);
  const exprt value =
    x.is_symbol() || x.is_constant() ? x : fresh("$math_arg$", x);
  if (value.type() == double_type())
    return value;
  return typecast_exprt(value, double_type());
}

exprt bounded_math::within(double low, double high)
{
  const exprt result = fresh("$math$", nil_exprt());
  assume(between(result, low, high));
  return result;
}

void bounded_math::raise_math_errors(
  const std::string &function,
  const exprt &x,
  const exprt &value)
{
  raise_if(domain_error(function, x), "ValueError", "math domain error");
  raise_if(
    and_exprt(
      not_exprt(or_exprt(is_nan(x), is_infinite(x))), is_infinite(value)),
    "OverflowError",
    "math range error");
}

void bounded_math::raise_if(
  const exprt &condition,
  const std::string &exception,
  const std::string &message)
{
  const locationt location = converter_.get_location_from_decl(call_);
  exprt raise =
    converter_.get_exception_handler().gen_exception_raise(exception, message);
  raise.location() = location;
  raise.location().user_provided(true);
  code_expressiont raise_code(raise);
  raise_code.location() = location;
  code_ifthenelset guard;
  guard.cond() = condition;
  guard.then_case() = raise_code;
  guard.location() = location;
  guard.location().property("skipped");
  converter_.add_instruction(guard);
}

void bounded_math::assume(const exprt &condition)
{
  code_assumet assumption(condition);
  assumption.location() = converter_.get_location_from_decl(call_);
  converter_.add_instruction(assumption);
}
