#pragma once

#include <nlohmann/json.hpp>
#include <util/irep/expr.h>
#include <util/irep/location.h>
#include <util/irep/std_code.h>
#include <util/irep/std_expr.h>

#include <map>
#include <set>
#include <string>
#include <vector>

class python_converter;

/// `np.random.<name>(...)` as a nondeterministic over-approximation
/// (ADR-NP-002): each sample is a fresh symbol constrained to the support of
/// its distribution. Probabilities, uniformity and generator state are not
/// modelled. Arguments are evaluated once, in source order, and validated
/// before any sample is drawn.
class numpy_random
{
public:
  numpy_random(python_converter &converter, const nlohmann::json &call);

  static bool
  is_random_call(const nlohmann::json &call, const nlohmann::json &ast);

  exprt get();

private:
  python_converter &converter_;
  const nlohmann::json &call_;
  /// The function as spelled, and the distribution it samples.
  std::string function_;
  std::string name_;
  locationt location_;
  code_blockt block_;
  /// Bound parameter values, each read once into a temporary.
  std::map<std::string, exprt> values_;
  std::string dtype_;
  std::set<std::string> bound_;
  /// The first binding error, raised as a TypeError once every argument has
  /// been evaluated.
  std::string error_;
  /// The literal output shape when sized_, a scalar otherwise.
  std::vector<long long> shape_;
  bool sized_ = false;
  static constexpr long long max_samples = 1024;

  void bind_arguments();
  void bind_positional();
  void bind_keywords();
  void bind(const std::string &param, const nlohmann::json &arg);
  [[noreturn]] void reject_unpacked() const;
  std::string missing_argument() const;
  bool takes_dimensions() const;
  void bind_dimensions();
  void read_shape(const nlohmann::json &size);
  [[noreturn]] void reject_shape() const;
  void check_shape();
  exprt sample_array();
  void fill(const exprt &target);
  exprt evaluate_once(const nlohmann::json &arg, const std::string &param);
  exprt value(const std::string &param) const;

  void validate();
  void validate_randint();
  exprt sample();
  exprt sample_integer();

  /// A new temporary holding @p value, or a nondeterministic one.
  exprt fresh(const typet &type, const exprt &value = nil_exprt());
  void assume(const exprt &cond);
  void
  raise_if(const exprt &cond, const std::string &exc, const std::string &msg);
  exprt finish(const exprt &result);
};
