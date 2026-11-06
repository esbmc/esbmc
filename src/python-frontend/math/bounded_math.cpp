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

#include <cmath>
#include <numbers>
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
  static const std::set<std::string> one = {"exp", "exp2", "cos", "cosh"};
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
  return is_bounded(function) || function == "sqrt";
}

double bounded_math::host_value(const std::string &function, double x)
{
  return transcendental::host_value(function, x);
}

bool bounded_math::outside_domain(const std::string &function, double x)
{
  const transcendental::domain domain = transcendental::domain_of(function);
  return !(x >= domain.low && x <= domain.high);
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
  assume(or_exprt(not_exprt(regular), between(result, larger, upper)));
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

// The special values of C99 F.9.1.4 are exact; otherwise the result lies in
// the quadrant of (x, y), on the side of pi/4 or 3*pi/4 that |y| >= |x| picks.
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
  assume(or_exprt(not_exprt(regular), in_quadrant));
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

exprt bounded_math::read_once(const exprt &x)
{
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
