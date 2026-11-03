#include <util/arith/transcendental.h>
#include <util/config/config.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <map>
#include <numbers>
#include <set>
#include <stdexcept>

using namespace transcendental;

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
  static const double pi = std::numbers::pi, half_pi = pi / 2;
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

// How many ulps a result may lie from the C library's value
// (--python-math-ulps): 8 by default, as NumPy's vectorised code differs
// from libm (SVML on AVX-512 is tested to 4 ulps of the exact value), and 0
// for a correctly rounded library.
double max_ulps()
{
  const std::string option = config.options.get_option("python-math-ulps");
  if (option.empty())
    return 8;
  const int ulps = std::stoi(option);
  if (ulps < 0)
    throw std::runtime_error("--python-math-ulps must not be negative");
  return ulps;
}

double ulps(double value)
{
  return max_ulps() *
         (std::nextafter(std::fabs(value), inf) - std::fabs(value));
}

// The results around x: at x itself, or near it, where only a zero keeps
// its sign under a relative error. A nonzero result may underflow to zero
// but keeps its sign, which a few ulps of a subnormal would cross.
interval around(const std::string &function, double x, bool at_x)
{
  const double value = host_value(function, x);
  const bool exact = exact_everywhere(function, x, value);
  if (exact && (at_x || value == 0.0 || !std::isfinite(value)))
    return {value, value};
  const math_spec &spec = spec_of(function);
  const double low = value > 0 ? std::max(spec.range_low, 0.0) : spec.range_low;
  const double high =
    value < 0 ? std::min(spec.range_high, 0.0) : spec.range_high;
  return {
    std::max(value - ulps(value), low), std::min(value + ulps(value), high)};
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
    3.0,
    4.0,
    6.0,
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

std::vector<piece> build_pieces(const std::string &function)
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
        const double peak = std::round(host_value(function, turn));
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

} // namespace

bool transcendental::is_bounded(const std::string &function)
{
  return specs().count(function) != 0;
}

bool transcendental::is_odd(const std::string &function)
{
  static const std::set<std::string> odd = {
    "sin", "tan", "arctan", "arcsin", "sinh", "tanh"};
  return odd.count(function) != 0;
}

double transcendental::host_value(const std::string &function, double x)
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

interval transcendental::at(const std::string &function, double x)
{
  const double value = host_value(function, x);
  if (!is_bounded(function) || exact_everywhere(function, x, value))
    return {value, value};
  return around(function, x, true);
}

const std::vector<piece> &transcendental::pieces(const std::string &function)
{
  static std::map<std::string, std::vector<piece>> cache;
  auto it = cache.find(function);
  if (it == cache.end())
    it = cache.emplace(function, build_pieces(function)).first;
  return it->second;
}

domain transcendental::domain_of(const std::string &function)
{
  const math_spec &spec = spec_of(function);
  return {spec.domain_low, spec.domain_high, spec.periodic};
}
