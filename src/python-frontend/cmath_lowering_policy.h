#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

class exprt;

namespace cmath_lowering_policy
{
struct cost_budgett
{
  std::size_t max_nodes = 64;
  std::size_t max_depth = 12;
  std::size_t max_expensive_ops = 8;
  std::size_t max_nonlinear_ops = 12;
};

struct cost_metricst
{
  std::size_t nodes = 0;
  std::size_t depth = 0;
  std::size_t expensive_ops = 0;
  std::size_t nonlinear_ops = 0;
  bool over_budget = false;
};

bool is_inverse_function(const std::string &func_name);

bool is_structural_zero(const exprt &expr);
bool is_structural_one(const exprt &expr);

bool is_domain_safe_pure_imag_fastpath(
  const std::string &func_name,
  const exprt &imag_expr);

cost_budgett budget_for_function(const std::string &func_name);

cost_metricst estimate_expr_cost(
  const exprt &expr,
  const cost_budgett &budget = cost_budgett());

bool is_within_budget(
  const exprt &expr,
  const cost_budgett &budget = cost_budgett());

bool is_within_budget(const std::string &func_name, const exprt &expr);

enum class fallback_reasont : std::uint8_t
{
  generic = 0,
  budget_exceeded_log10,
  budget_exceeded_log_unary,
  budget_exceeded_log_base,
  budget_exceeded_inverse,
  nonzero_real_axis,
  domain_guard_failed
};

struct telemetry_snapshott
{
  std::size_t fast_path_hits = 0;
  std::size_t model_fallbacks = 0;
  std::size_t budget_bailouts = 0;
};

struct function_telemetryt
{
  std::size_t fast_path_hits = 0;
  std::size_t model_fallbacks = 0;
  std::size_t budget_bailouts = 0;
  std::unordered_map<std::uint8_t, std::size_t> fallback_reasons;
};

void record_fast_path_hit(const std::string &func_name);
void record_model_fallback(
  const std::string &func_name,
  bool budget_bailout,
  fallback_reasont reason = fallback_reasont::generic);
telemetry_snapshott get_telemetry_snapshot();
std::unordered_map<std::string, function_telemetryt>
get_function_telemetry_snapshot();
void log_telemetry_if_verbose();
const char *fallback_reason_to_string(fallback_reasont reason);
} // namespace cmath_lowering_policy
