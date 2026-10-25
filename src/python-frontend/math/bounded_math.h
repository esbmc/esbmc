#pragma once

#include <nlohmann/json.hpp>
#include <util/irep/expr.h>

#include <string>

class python_converter;

/// The transcendental functions shared by numpy.* and math.* (NumPy
/// spelling): any double within 8 ulps of the C library's value and inside
/// the function's range, exact where every implementation is (ADR-NP-005).
class bounded_math
{
public:
  bounded_math(python_converter &converter, const nlohmann::json &call);

  /// Functions whose result of a run-time argument is bounded.
  static bool is_bounded(const std::string &function);
  /// is_bounded() or exact (sqrt).
  static bool has_runtime_model(const std::string &function);
  static double host_value(const std::string &function, double x);

  /// The result for a literal argument.
  exprt literal(const std::string &function, double x);
  /// The result for a double known only at run time, which is read once.
  exprt runtime(const std::string &function, const exprt &x);

private:
  python_converter &converter_;
  const nlohmann::json &call_;

  exprt
  special_cases(const std::string &function, const exprt &x, exprt result);
  exprt fresh_double(const std::string &name, const exprt &value);
  exprt within(double low, double high);
  void assume(const exprt &condition);
};
