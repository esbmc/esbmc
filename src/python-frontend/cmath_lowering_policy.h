#pragma once

#include <nlohmann/json.hpp>
#include <cstddef>

/// Budget model that decides when cmath calls should be lowered inline
/// vs dispatched to the Python model.  Inline expansion of complex-math
/// (e.g. cmath.log) can create large IR trees that stall the solver;
/// the budget caps structural cost of the argument AST to prevent that.
namespace cmath_lowering_policy
{

struct expr_cost
{
  size_t node_count = 0;
  size_t depth = 0;
  size_t call_count = 0;
  size_t binop_count = 0;
};

struct budget
{
  size_t max_node_count = 32;
  size_t max_depth = 6;
  size_t max_call_count = 2;
  size_t max_binop_count = 8;
};

expr_cost measure(const nlohmann::json &arg);
bool within_budget(const nlohmann::json &arg, const budget &b = {});

} // namespace cmath_lowering_policy
