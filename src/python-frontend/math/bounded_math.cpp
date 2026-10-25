#include <python-frontend/math/bounded_math.h>
#include <python-frontend/python_converter.h>
#include <irep2/irep2_utils.h>
#include <util/arith/arith_tools.h>
#include <util/irep/migrate.h>
#include <util/lang/c_types.h>
#include <util/irep/std_code.h>
#include <util/irep/std_expr.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <vector>

namespace
{
const double inf = INFINITY;

// Where a function is bounded piecewise between breakpoints, its range, the
// nearest doubles to where it turns or jumps, and the domain outside which
// the result is NaN. Outside [from, to] a periodic function only keeps its
// range for a finite argument.
struct math_spec
{
  double from, to;
  double range_low, range_high;
  std::vector<double> turning;
  bool periodic;
  std::vector<double> poles = {};
  double domain_low = -inf, domain_high = inf;
};

const std::map<std::string, math_spec> &specs()
{
  static const double pi = M_PI, half_pi = M_PI_2;
  static const std::map<std::string, math_spec> table = {
    {"exp", {-inf, inf, 0.0, inf, {}, false}},
    {"log", {DBL_TRUE_MIN, inf, -inf, inf, {}, false, {}, 0.0}},
    {"log2", {DBL_TRUE_MIN, inf, -inf, inf, {}, false, {}, 0.0}},
    {"log10", {DBL_TRUE_MIN, inf, -inf, inf, {}, false, {}, 0.0}},
    {"sin",
     {-8.0,
      8.0,
      -1.0,
      1.0,
      {-5 * half_pi, -3 * half_pi, -half_pi, half_pi, 3 * half_pi, 5 * half_pi},
      true}},
    {"cos", {-8.0, 8.0, -1.0, 1.0, {-2 * pi, -pi, 0.0, pi, 2 * pi}, true}},
    {"tan", {-4.5, 4.5, -DBL_MAX, DBL_MAX, {}, true, {-half_pi, half_pi}}},
    {"arcsin", {-1.0, 1.0, -half_pi, half_pi, {}, false, {}, -1.0, 1.0}},
    {"arccos", {-1.0, 1.0, 0.0, pi, {}, false, {}, -1.0, 1.0}},
    {"arctan", {-inf, inf, -half_pi, half_pi, {}, false}},
    {"sinh", {-inf, inf, -inf, inf, {}, false}},
    {"cosh", {-inf, inf, 1.0, inf, {0.0}, false}},
    {"tanh", {-inf, inf, -1.0, 1.0, {}, false}}};
  return table;
}

const math_spec &spec_of(const std::string &function)
{
  auto it = specs().find(function);
  if (it == specs().end())
    throw std::runtime_error("Unsupported Numpy unary function: " + function);
  return it->second;
}

bool is_odd(const std::string &function)
{
  static const std::set<std::string> odd = {
    "sin", "tan", "arctan", "arcsin", "sinh", "tanh"};
  return odd.count(function) != 0;
}

// Exact for every implementation: a correctly rounded square root, an odd
// function at a signed zero, exp(0), log(1), log2 of a power of two, a limit
// at infinity, and any result that is not finite.
bool exact_everywhere(const std::string &function, double x, double value)
{
  static const std::set<std::string> one_at_zero = {"exp", "cos", "cosh"};
  static const std::set<std::string> zero_at_one = {
    "log", "log2", "log10", "arccos"};
  int exponent = 0;
  const bool power_of_two = x > 0 && std::frexp(x, &exponent) == 0.5;
  return function == "sqrt" || function == "rint" || !std::isfinite(value) ||
         !std::isfinite(x) ||
         (x == 0.0 && (is_odd(function) || one_at_zero.count(function) != 0)) ||
         (x == 1.0 && zero_at_one.count(function) != 0) ||
         (function == "log2" && power_of_two);
}

double ulps(double value)
{
  return 8 * (std::nextafter(std::fabs(value), inf) - std::fabs(value));
}

struct interval
{
  double low, high;
};

// The results around x: at x itself, or near it, where only a zero keeps
// its sign under a relative error.
interval around(const std::string &function, double x, bool at_x)
{
  const double value = bounded_math::host_value(function, x);
  const bool exact = exact_everywhere(function, x, value);
  if (exact && (at_x || value == 0.0 || !std::isfinite(value)))
    return {value, value};
  const math_spec &spec = spec_of(function);
  return {
    std::max(value - ulps(value), spec.range_low),
    std::min(value + ulps(value), spec.range_high)};
}

// tan grows towards its pole, so the pole lies past the nearest double on
// the side where tan has that double's sign.
double pole_partner(double pole)
{
  return std::nextafter(pole, std::tan(pole) > 0 ? inf : -inf);
}

bool spans_pole(const math_spec &spec, double a, double b)
{
  for (double pole : spec.poles)
    if (
      std::min(pole, pole_partner(pole)) == a &&
      std::max(pole, pole_partner(pole)) == b)
      return true;
  return false;
}

std::vector<double> breakpoints(const math_spec &spec)
{
  static const std::vector<double> grid = {
    0.0,
    0x1p-26,
    0.125,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    32.0,
    256.0,
    709.78,
    709.79,
    710.47,
    710.48,
    1e300,
    DBL_MIN,
    DBL_TRUE_MIN};
  std::set<double> points = {spec.from, spec.to};
  for (double magnitude : grid)
    for (double point : {magnitude, -magnitude})
      if (point > spec.from && point < spec.to)
        points.insert(point);
  points.insert(spec.turning.begin(), spec.turning.end());
  for (double pole : spec.poles)
    points.insert({pole, pole_partner(pole)});
  return {points.begin(), points.end()};
}

struct piece
{
  double from, to;
  interval result;
};

std::vector<piece> pieces(const std::string &function)
{
  const math_spec &spec = spec_of(function);
  const std::vector<double> points = breakpoints(spec);
  std::vector<piece> out;
  for (std::size_t i = 0; i + 1 < points.size(); ++i)
  {
    const double a = points[i], b = points[i + 1];
    if (spans_pole(spec, a, b))
      continue;
    const interval left = around(function, a, false);
    const interval right = around(function, b, false);
    interval result = {
      std::min(left.low, right.low), std::max(left.high, right.high)};
    for (double turn : spec.turning)
      if (std::nextafter(turn, -inf) <= b && std::nextafter(turn, inf) >= a)
      {
        const double peak =
          std::round(bounded_math::host_value(function, turn));
        result = {std::min(result.low, peak), std::max(result.high, peak)};
      }
    out.push_back({a, b, result});
  }
  if (spec.periodic)
  {
    const interval range = {spec.range_low, spec.range_high};
    out.push_back({-DBL_MAX, spec.from, range});
    out.push_back({spec.to, DBL_MAX, range});
  }
  return out;
}

exprt between(const exprt &value, double low, double high)
{
  return and_exprt(
    binary_relation_exprt(value, ">=", from_double(low, double_type())),
    binary_relation_exprt(value, "<=", from_double(high, double_type())));
}

// An exact value at a signed zero, where the bounds cannot keep the sign.
exprt at_zero(const std::string &function, const exprt &x)
{
  static const std::set<std::string> one = {"exp", "cos", "cosh"};
  static const std::set<std::string> pole = {"log", "log2", "log10"};
  if (one.count(function) != 0)
    return from_double(1.0, double_type());
  if (pole.count(function) != 0)
    return from_double(-inf, double_type());
  if (is_odd(function))
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
} // namespace

bounded_math::bounded_math(
  python_converter &converter,
  const nlohmann::json &call)
  : converter_(converter), call_(call)
{
}

bool bounded_math::is_bounded(const std::string &function)
{
  return specs().count(function) != 0;
}

bool bounded_math::has_runtime_model(const std::string &function)
{
  return is_bounded(function) || function == "sqrt";
}

double bounded_math::host_value(const std::string &function, double x)
{
  static const std::map<std::string, double (*)(double)> functions = {
    {"sin", [](double v) { return std::sin(v); }},
    {"cos", [](double v) { return std::cos(v); }},
    {"tan", [](double v) { return std::tan(v); }},
    {"exp", [](double v) { return std::exp(v); }},
    {"log", [](double v) { return std::log(v); }},
    {"log2", [](double v) { return std::log2(v); }},
    {"log10", [](double v) { return std::log10(v); }},
    {"sqrt", [](double v) { return std::sqrt(v); }},
    {"arcsin", [](double v) { return std::asin(v); }},
    {"arccos", [](double v) { return std::acos(v); }},
    {"arctan", [](double v) { return std::atan(v); }},
    {"sinh", [](double v) { return std::sinh(v); }},
    {"cosh", [](double v) { return std::cosh(v); }},
    {"tanh", [](double v) { return std::tanh(v); }},
    {"rint", [](double v) { return std::rint(v); }}};
  auto it = functions.find(function);
  if (it == functions.end())
    throw std::runtime_error("Unsupported Numpy unary function: " + function);
  return it->second(x);
}

exprt bounded_math::literal(const std::string &function, double x)
{
  const double value = host_value(function, x);
  if (!is_bounded(function) || exact_everywhere(function, x, value))
    return from_double(value, double_type());
  const interval result = around(function, x, true);
  return within(result.low, result.high);
}

exprt bounded_math::runtime(const std::string &function, const exprt &x)
{
  if (function == "sqrt")
    return converter_.get_math_handler().handle_sqrt(x, call_);
  const exprt argument =
    x.is_symbol() || x.is_constant() ? x : fresh_double("$math_arg$", x);
  const exprt result = fresh_double("$math$", nil_exprt());
  exprt::operandst bounds;
  for (const piece &p : pieces(function))
    bounds.push_back(or_exprt(
      not_exprt(between(argument, p.from, p.to)),
      between(result, p.result.low, p.result.high)));
  assume(and_exprt(bounds));
  return special_cases(function, argument, result);
}

// NaN propagates, an argument outside the domain gives NaN, and a signed
// zero or an infinity has an exact result.
exprt bounded_math::special_cases(
  const std::string &function,
  const exprt &x,
  exprt result)
{
  const math_spec &spec = spec_of(function);
  const exprt zero = at_zero(function, x);
  if (!zero.is_nil())
    result = if_exprt(
      equality_exprt(x, from_double(0.0, double_type())), zero, result);
  const exprt positive =
    binary_relation_exprt(x, ">", from_double(0.0, double_type()));
  result = if_exprt(
    is_infinite(x),
    if_exprt(positive, literal(function, inf), literal(function, -inf)),
    result);
  result = if_exprt(
    not_exprt(between(x, spec.domain_low, spec.domain_high)),
    from_double(NAN, double_type()),
    result);
  return if_exprt(is_nan(x), x, result);
}

exprt bounded_math::fresh_double(const std::string &name, const exprt &value)
{
  symbolt &tmp =
    converter_.create_tmp_symbol(call_, name, double_type(), exprt());
  const exprt symbol = symbol_expr(tmp);
  const locationt location = converter_.get_location_from_decl(call_);
  code_declt decl(symbol);
  decl.location() = location;
  converter_.add_instruction(decl);
  code_assignt assign(
    symbol, value.is_nil() ? side_effect_expr_nondett(double_type()) : value);
  assign.location() = location;
  converter_.add_instruction(assign);
  return symbol;
}

exprt bounded_math::within(double low, double high)
{
  const exprt result = fresh_double("$math$", nil_exprt());
  assume(between(result, low, high));
  return result;
}

void bounded_math::assume(const exprt &condition)
{
  code_assumet assumption(condition);
  assumption.location() = converter_.get_location_from_decl(call_);
  converter_.add_instruction(assumption);
}
