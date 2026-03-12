#include <python-frontend/cmath_lowering_policy.h>

#include <util/config.h>
#include <util/irep.h>
#include <util/ieee_float.h>
#include <util/message.h>
#include <util/std_expr.h>

#include <array>
#include <atomic>
#include <cmath>
#include <unordered_map>

namespace
{
std::atomic<std::size_t> g_fast_path_hits{0};
std::atomic<std::size_t> g_model_fallbacks{0};
std::atomic<std::size_t> g_budget_bailouts{0};

enum class solver_profilet : std::uint8_t
{
  default_solver = 0,
  linear_solver,
  general_solver
};

enum class telemetry_buckett : std::size_t
{
  asin = 0,
  atan,
  asinh,
  atanh,
  log,
  log10,
  other,
  count
};

constexpr std::size_t bucket_count =
  static_cast<std::size_t>(telemetry_buckett::count);
constexpr std::size_t reason_count =
  static_cast<std::size_t>(
    cmath_lowering_policy::fallback_reasont::domain_guard_failed) +
  1;

const std::array<const char *, bucket_count> bucket_names =
  {"asin", "atan", "asinh", "atanh", "log", "log10", "other"};

std::array<std::atomic<std::size_t>, bucket_count> g_fast_path_hits_by_bucket{};
std::array<std::atomic<std::size_t>, bucket_count>
  g_model_fallbacks_by_bucket{};
std::array<std::atomic<std::size_t>, bucket_count>
  g_budget_bailouts_by_bucket{};
std::array<std::array<std::atomic<std::size_t>, reason_count>, bucket_count>
  g_fallback_reasons_by_bucket{};

telemetry_buckett telemetry_bucket_for_function(const std::string &func_name)
{
  if (func_name == "asin")
    return telemetry_buckett::asin;
  if (func_name == "atan")
    return telemetry_buckett::atan;
  if (func_name == "asinh")
    return telemetry_buckett::asinh;
  if (func_name == "atanh")
    return telemetry_buckett::atanh;
  if (func_name == "log")
    return telemetry_buckett::log;
  if (func_name == "log10")
    return telemetry_buckett::log10;
  return telemetry_buckett::other;
}

solver_profilet solver_profile()
{
  static const solver_profilet cached_profile = []() {
    if (
      config.options.get_bool_option("boolector") ||
      config.options.get_bool_option("bitwuzla"))
      return solver_profilet::linear_solver;

    if (
      config.options.get_bool_option("z3") ||
      config.options.get_bool_option("z3-debug"))
      return solver_profilet::general_solver;

    return solver_profilet::default_solver;
  }();
  return cached_profile;
}

bool try_get_constant_float(const exprt &expr, double &out)
{
  if (expr.id() == "typecast" && !expr.operands().empty())
    return try_get_constant_float(expr.op0(), out);

  if (!expr.is_constant() || !expr.type().is_floatbv())
    return false;

  ieee_floatt value(to_constant_expr(expr));
  out = value.to_double();
  return true;
}

void estimate_cost_recursive(
  const exprt &expr,
  const std::size_t depth,
  cmath_lowering_policy::cost_metricst &cost)
{
  cost.nodes++;
  if (depth > cost.depth)
    cost.depth = depth;

  const irep_idt &id = expr.id();
  if (
    id == "ieee_div" || id == "ieee_sqrt" || id == "ieee_log" ||
    id == "ieee_pow" || id == "ieee_rem")
  {
    cost.expensive_ops++;
  }
  else if (id == "sideeffect" && expr.statement() == "function_call")
  {
    cost.expensive_ops += 2;
  }

  // Penalize nonlinear arithmetic to avoid lowering formulas that tend to
  // explode in SMT encoding.
  if (id == "ieee_pow")
  {
    cost.nonlinear_ops += 2;
  }
  else if (
    (id == "ieee_mul" || id == "ieee_div") && expr.operands().size() >= 2 &&
    (!expr.op0().is_constant() || !expr.op1().is_constant()))
  {
    cost.nonlinear_ops++;
  }

  for (const auto &operand : expr.operands())
    estimate_cost_recursive(operand, depth + 1, cost);
}
} // namespace

namespace cmath_lowering_policy
{
bool is_inverse_function(const std::string &func_name)
{
  return (
    func_name == "asin" || func_name == "atan" || func_name == "asinh" ||
    func_name == "atanh");
}

bool is_structural_zero(const exprt &expr)
{
  double value = 0.0;
  if (!try_get_constant_float(expr, value))
    return false;
  return value == 0.0;
}

bool is_structural_one(const exprt &expr)
{
  double value = 0.0;
  if (!try_get_constant_float(expr, value))
    return false;
  return value == 1.0;
}

bool is_domain_safe_pure_imag_fastpath(
  const std::string &func_name,
  const exprt &imag_expr)
{
  if (func_name == "asin" || func_name == "atanh")
    return true;

  double imag_value = 0.0;
  if (!try_get_constant_float(imag_expr, imag_value))
    return false;
  if (!std::isfinite(imag_value))
    return false;

  const double abs_imag = std::fabs(imag_value);
  if (func_name == "atan")
    return abs_imag < 1.0;
  if (func_name == "asinh")
    return abs_imag <= 1.0;
  return false;
}

cost_budgett budget_for_function(const std::string &func_name)
{
  const solver_profilet profile = solver_profile();

  if (func_name == "log" || func_name == "log10")
  {
    if (profile == solver_profilet::linear_solver)
      return cost_budgett{44, 9, 5, 7};
    if (profile == solver_profilet::general_solver)
      return cost_budgett{52, 11, 7, 9};
    return cost_budgett{48, 10, 6, 8};
  }

  if (func_name == "asin" || func_name == "atanh")
  {
    if (profile == solver_profilet::linear_solver)
      return cost_budgett{58, 11, 7, 9};
    if (profile == solver_profilet::general_solver)
      return cost_budgett{68, 13, 9, 11};
    return cost_budgett{64, 12, 8, 10};
  }

  if (func_name == "atan" || func_name == "asinh")
  {
    if (profile == solver_profilet::linear_solver)
      return cost_budgett{52, 10, 6, 8};
    if (profile == solver_profilet::general_solver)
      return cost_budgett{60, 12, 8, 10};
    return cost_budgett{56, 11, 7, 9};
  }

  return cost_budgett{};
}

cost_metricst estimate_expr_cost(const exprt &expr, const cost_budgett &budget)
{
  using cache_keyt = exprt;
  using cachet =
    std::unordered_map<cache_keyt, cost_metricst, irep_full_hash, irep_full_eq>;
  static thread_local cachet metrics_cache;

  if (auto it = metrics_cache.find(expr); it != metrics_cache.end())
  {
    cost_metricst cached = it->second;
    cached.over_budget = cached.nodes > budget.max_nodes ||
                         cached.depth > budget.max_depth ||
                         cached.expensive_ops > budget.max_expensive_ops ||
                         cached.nonlinear_ops > budget.max_nonlinear_ops;
    return cached;
  }

  cost_metricst cost;
  estimate_cost_recursive(expr, 1, cost);
  if (metrics_cache.size() > 4096)
    metrics_cache.clear();
  metrics_cache.emplace(expr, cost);

  cost.over_budget = cost.nodes > budget.max_nodes ||
                     cost.depth > budget.max_depth ||
                     cost.expensive_ops > budget.max_expensive_ops ||
                     cost.nonlinear_ops > budget.max_nonlinear_ops;
  return cost;
}

bool is_within_budget(const exprt &expr, const cost_budgett &budget)
{
  return !estimate_expr_cost(expr, budget).over_budget;
}

bool is_within_budget(const std::string &func_name, const exprt &expr)
{
  const cost_budgett budget = budget_for_function(func_name);
  const cost_metricst cost = estimate_expr_cost(expr, budget);
  return !cost.over_budget;
}

const char *fallback_reason_to_string(fallback_reasont reason)
{
  switch (reason)
  {
  case fallback_reasont::generic:
    return "generic";
  case fallback_reasont::budget_exceeded_log10:
    return "budget_exceeded_log10";
  case fallback_reasont::budget_exceeded_log_unary:
    return "budget_exceeded_log_unary";
  case fallback_reasont::budget_exceeded_log_base:
    return "budget_exceeded_log_base";
  case fallback_reasont::budget_exceeded_inverse:
    return "budget_exceeded_inverse";
  case fallback_reasont::nonzero_real_axis:
    return "nonzero_real_axis";
  case fallback_reasont::domain_guard_failed:
    return "domain_guard_failed";
  }
  return "unknown";
}

void record_fast_path_hit(const std::string &func_name)
{
  ++g_fast_path_hits;
  const std::size_t bucket =
    static_cast<std::size_t>(telemetry_bucket_for_function(func_name));
  g_fast_path_hits_by_bucket[bucket].fetch_add(1, std::memory_order_relaxed);
}

void record_model_fallback(
  const std::string &func_name,
  bool budget_bailout,
  fallback_reasont reason)
{
  ++g_model_fallbacks;
  if (budget_bailout)
    ++g_budget_bailouts;

  const std::size_t bucket =
    static_cast<std::size_t>(telemetry_bucket_for_function(func_name));
  g_model_fallbacks_by_bucket[bucket].fetch_add(1, std::memory_order_relaxed);
  if (budget_bailout)
    g_budget_bailouts_by_bucket[bucket].fetch_add(1, std::memory_order_relaxed);

  const std::size_t reason_index = static_cast<std::size_t>(reason);
  if (reason_index < reason_count)
  {
    g_fallback_reasons_by_bucket[bucket][reason_index].fetch_add(
      1, std::memory_order_relaxed);
  }
}

telemetry_snapshott get_telemetry_snapshot()
{
  telemetry_snapshott snapshot;
  snapshot.fast_path_hits = g_fast_path_hits.load();
  snapshot.model_fallbacks = g_model_fallbacks.load();
  snapshot.budget_bailouts = g_budget_bailouts.load();
  return snapshot;
}

std::unordered_map<std::string, function_telemetryt>
get_function_telemetry_snapshot()
{
  std::unordered_map<std::string, function_telemetryt> snapshot;
  for (std::size_t bucket = 0; bucket < bucket_count; ++bucket)
  {
    function_telemetryt stats;
    stats.fast_path_hits =
      g_fast_path_hits_by_bucket[bucket].load(std::memory_order_relaxed);
    stats.model_fallbacks =
      g_model_fallbacks_by_bucket[bucket].load(std::memory_order_relaxed);
    stats.budget_bailouts =
      g_budget_bailouts_by_bucket[bucket].load(std::memory_order_relaxed);

    for (std::size_t reason_index = 0; reason_index < reason_count;
         ++reason_index)
    {
      const std::size_t count =
        g_fallback_reasons_by_bucket[bucket][reason_index].load(
          std::memory_order_relaxed);
      if (count != 0)
        stats.fallback_reasons[static_cast<std::uint8_t>(reason_index)] = count;
    }

    if (
      stats.fast_path_hits == 0 && stats.model_fallbacks == 0 &&
      stats.budget_bailouts == 0 && stats.fallback_reasons.empty())
      continue;

    snapshot.emplace(bucket_names[bucket], std::move(stats));
  }
  return snapshot;
}

void log_telemetry_if_verbose()
{
  // Emit only in debug verbosity to avoid noisy default output.
  if (!messaget::state.target("python", VerbosityLevel::Debug))
    return;

  const telemetry_snapshott total = get_telemetry_snapshot();
  log_debug(
    "python",
    "cmath-lowering telemetry total: fast_path_hits={}, model_fallbacks={}, "
    "budget_bailouts={}",
    total.fast_path_hits,
    total.model_fallbacks,
    total.budget_bailouts);

  const auto per_func = get_function_telemetry_snapshot();
  for (const auto &entry : per_func)
  {
    log_debug(
      "python",
      "cmath-lowering telemetry [{}]: fast_path_hits={}, model_fallbacks={}, "
      "budget_bailouts={}",
      entry.first,
      entry.second.fast_path_hits,
      entry.second.model_fallbacks,
      entry.second.budget_bailouts);

    for (const auto &reason_entry : entry.second.fallback_reasons)
    {
      log_debug(
        "python",
        "cmath-lowering telemetry [{}] fallback_reason [{}] => {}",
        entry.first,
        fallback_reason_to_string(
          static_cast<fallback_reasont>(reason_entry.first)),
        reason_entry.second);
    }
  }
}
} // namespace cmath_lowering_policy
