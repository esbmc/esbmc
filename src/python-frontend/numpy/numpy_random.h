#pragma once

#include <nlohmann/json.hpp>
#include <util/irep/expr.h>
#include <util/irep/location.h>
#include <util/irep/std_code.h>
#include <util/irep/std_expr.h>
#include <util/irep/std_types.h>

#include <map>
#include <optional>
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

  /// The literal output shape of a random call that returns an array, read
  /// from the source; nullopt for a scalar result or a shape it rejects.
  static std::optional<std::vector<std::size_t>>
  output_shape(const nlohmann::json &call, const nlohmann::json &ast);

  /// "list" for a random call that returns an array, so the annotator types
  /// it and arithmetic on it as it does a numpy constructor's result; empty
  /// for a scalar.
  static std::string
  list_annotation(const nlohmann::json &call, const nlohmann::json &ast);

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
  typet element_type_;
  /// choice(): the population as a numpy array, and the indices drawn so far.
  exprt population_;
  bool replace_ = true;
  bool has_probabilities_ = false;
  std::vector<exprt> drawn_indices_;

  exprt reject_unmodelled();
  void bind_arguments();
  void bind_positional();
  void bind_keywords();
  void bind(const std::string &param, const nlohmann::json &arg);
  [[noreturn]] void reject_unpacked() const;
  std::string missing_argument() const;
  exprt read_population(const nlohmann::json &arg);
  exprt materialize_population(const nlohmann::json &elements);
  [[noreturn]] void reject_population() const;
  bool holds_numpy_array(const nlohmann::json &arg) const;
  bool read_replace(const nlohmann::json &arg) const;
  bool read_probabilities(const nlohmann::json &arg);
  long long population_size() const;
  long long sample_count() const;
  void validate_choice();
  exprt sample_choice();
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
  exprt sample_real();
  exprt nan_from_parameters(const exprt &sample);

  /// A new temporary holding @p value, or a nondeterministic one.
  exprt fresh(const typet &type, const exprt &value = nil_exprt());
  void assume(const exprt &cond);
  void
  raise_if(const exprt &cond, const std::string &exc, const std::string &msg);
  exprt finish(const exprt &result);
};
