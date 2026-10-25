#include <python-frontend/math/bounded_math.h>
#include <python-frontend/python_converter.h>
#include <util/arith/arith_tools.h>
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
// Where a function is bounded piecewise between breakpoints, its range, and
// the nearest doubles to where it turns. Outside [from, to] a periodic
// function only keeps its range for a finite argument.
struct math_spec
{
  double from, to;
  double range_low, range_high;
  std::vector<double> turning;
  bool periodic;
};

const double inf = INFINITY;

const std::map<std::string, math_spec> &specs()
{
  static const double pi = M_PI, half_pi = M_PI_2;
  static const std::map<std::string, math_spec> table = {
    {"exp", {-inf, inf, 0.0, inf, {}, false}},
    {"log", {DBL_TRUE_MIN, inf, -inf, inf, {}, false}},
    {"log2", {DBL_TRUE_MIN, inf, -inf, inf, {}, false}},
    {"log10", {DBL_TRUE_MIN, inf, -inf, inf, {}, false}},
    {"sin",
     {-8.0,
      8.0,
      -1.0,
      1.0,
      {-5 * half_pi, -3 * half_pi, -half_pi, half_pi, 3 * half_pi, 5 * half_pi},
      true}},
    {"cos", {-8.0, 8.0, -1.0, 1.0, {-2 * pi, -pi, 0.0, pi, 2 * pi}, true}},
    {"tan", {-half_pi, half_pi, -DBL_MAX, DBL_MAX, {}, true}},
    {"arcsin", {-1.0, 1.0, -half_pi, half_pi, {}, false}},
    {"arccos", {-1.0, 1.0, 0.0, pi, {}, false}},
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

// Exact for every implementation: a correctly rounded square root, an odd
// function at a signed zero, exp(0), log(1), log2 of a power of two, and
// any result that is not finite.
bool exact_everywhere(const std::string &function, double x, double value)
{
  static const std::set<std::string> odd = {
    "sin", "tan", "arctan", "arcsin", "sinh", "tanh"};
  static const std::set<std::string> one_at_zero = {"exp", "cos", "cosh"};
  static const std::set<std::string> zero_at_one = {
    "log", "log2", "log10", "arccos"};
  int exponent = 0;
  const bool power_of_two = x > 0 && std::frexp(x, &exponent) == 0.5;
  return function == "sqrt" || function == "rint" || !std::isfinite(value) ||
         (x == 0.0 &&
          (odd.count(function) != 0 || one_at_zero.count(function) != 0)) ||
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
  if (one.count(function) != 0)
    return from_double(1.0, double_type());
  if (exact_everywhere(function, 0.0, 0.0))
    return x;
  return nil_exprt();
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
  const exprt argument =
    x.is_symbol() || x.is_constant() ? x : fresh_double("$math_arg$", x);
  const exprt result = fresh_double("$math$", nil_exprt());
  exprt::operandst bounds;
  for (const piece &p : pieces(function))
    bounds.push_back(or_exprt(
      not_exprt(between(argument, p.from, p.to)),
      between(result, p.result.low, p.result.high)));
  assume(and_exprt(bounds));
  const exprt zero = at_zero(function, argument);
  if (zero.is_nil())
    return result;
  return if_exprt(
    equality_exprt(argument, from_double(0.0, double_type())), zero, result);
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
