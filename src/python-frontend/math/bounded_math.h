#pragma once

#include <nlohmann/json.hpp>
#include <util/irep/expr.h>

#include <string>
#include <vector>

class python_converter;

/// numpy.* returns NaN or an infinity where math.* raises, and a literal
/// numpy.* argument gets the bounded result where math.* gets the C
/// library's exact value.
enum class math_module
{
  numpy,
  math
};

/// The transcendental functions shared by numpy.* and math.* (NumPy
/// spelling), bounded as util/arith/transcendental.h describes.
class bounded_math
{
public:
  bounded_math(python_converter &converter, const nlohmann::json &call);

  /// Functions whose result of a run-time argument is bounded.
  static bool is_bounded(const std::string &function);
  /// hypot and arctan2.
  static bool is_bounded_binary(const std::string &function);
  /// The numpy.* functions with a run-time model: is_bounded() but for the
  /// math-only gamma and lgamma, or exact (sqrt).
  static bool has_runtime_model(const std::string &function);
  static double host_value(const std::string &function, double x);
  /// x lies outside the domain, poles included, where math.* raises.
  static bool outside_domain(const std::string &function, double x);

  /// The numpy.* result for a literal argument.
  exprt literal(const std::string &function, double x);
  /// The result for a number known only at run time, which is read once.
  exprt
  runtime(const std::string &function, const exprt &x, math_module module);
  /// hypot(a, b), arctan2(a, b) or pow(a, b), each argument read once.
  /// NumPy and math.* agree on hypot and arctan2: neither raises, and an
  /// overflow is an infinity; math.pow raises Python's errors.
  exprt binary(
    const std::string &function,
    const exprt &a,
    const exprt &b,
    math_module module);
  /// math.hypot of any number of arguments, each read once: hypot() is 0
  /// and hypot(x) is |x|.
  exprt hypot_of(const std::vector<exprt> &args);
  /// The value of an argument at this point, before a later argument's
  /// call can change what it reads.
  exprt capture(const exprt &x);
  /// isnan, isinf or isfinite of a number, which is read once.
  static bool is_float_predicate(const std::string &function);
  exprt float_predicate(const std::string &function, const exprt &x);

private:
  python_converter &converter_;
  const nlohmann::json &call_;

  exprt
  bounded_call(const std::string &function, const exprt &x, math_module module);
  exprt
  special_cases(const std::string &function, const exprt &x, exprt result);
  void raise_math_errors(
    const std::string &function,
    const exprt &x,
    const exprt &value);
  exprt gamma_family(const std::string &function, const exprt &x);
  exprt power(const exprt &a, const exprt &b, math_module module);
  exprt constant_power(double base, double exponent, math_module module);
  exprt power_infinite_exponent(const exprt &x, const exprt &y);
  exprt power_infinite_base(const exprt &x, const exprt &y, const exprt &odd);
  exprt power_zero_base(const exprt &x, const exprt &y, const exprt &odd);
  exprt power_magnitude(const exprt &a, const exprt &y, const exprt &regular);
  exprt power_overflow(const exprt &a, const exprt &y, const exprt &m);
  exprt power_literal_exponent(const exprt &a, const exprt &y, const exprt &m);
  exprt relative_to(const exprt &product, const exprt &m, std::size_t steps);
  exprt hypot(const exprt &x, const exprt &y);
  exprt hypot_of_runtime(const std::vector<exprt> &args);
  exprt arctan2(const exprt &y, const exprt &x);
  exprt squares_relation(const exprt &x, const exprt &y, const exprt &r);
  exprt bounded_binary_call(
    const std::string &function,
    const exprt &a,
    const exprt &b);
  exprt fresh(const std::string &name, const exprt &value);
  exprt read_once(const exprt &x);
  exprt within(double low, double high);
  void raise_if(
    const exprt &condition,
    const std::string &exception,
    const std::string &message);
  void assume(const exprt &condition);
};
