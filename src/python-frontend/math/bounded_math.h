#pragma once

#include <nlohmann/json.hpp>
#include <util/irep/expr.h>

#include <string>

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
  /// is_bounded() or exact (sqrt).
  static bool has_runtime_model(const std::string &function);
  static double host_value(const std::string &function, double x);

  /// The numpy.* result for a literal argument.
  exprt literal(const std::string &function, double x);
  /// The result for a number known only at run time, which is read once.
  exprt
  runtime(const std::string &function, const exprt &x, math_module module);

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
  exprt fresh(const std::string &name, const exprt &value);
  exprt read_once(const exprt &x);
  exprt within(double low, double high);
  void raise_if(
    const exprt &condition,
    const std::string &exception,
    const std::string &message);
  void assume(const exprt &condition);
};
