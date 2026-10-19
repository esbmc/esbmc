#pragma once

#include <transition-system/transition_system.h>
#include <solvers/smt/smt_conv.h>

#include <cstdint>
#include <memory>

/// BMC over a transition system, unrolled step by step in a single solver.
/// Each step's violation is queried through its own activation literal.
class ts_enginet
{
public:
  /// Without `incremental`, the solver is never asked to track assumptions;
  /// only solve_prefix and solve_bound may be used.
  ts_enginet(
    const transition_systemt &ts,
    const namespacet &ns,
    optionst &options,
    bool incremental = true);

  /// Exit codes follow do_bmc_strategy: 0 SUCCESSFUL or UNKNOWN, 1 FAILED,
  /// 6 solver error.
  int bmc(uint64_t max_k, bool push_pop);

  /// One monolithic BMC query on a fresh engine: steps 0..k unrolled, the
  /// violation at step k asserted, a single check-sat.
  smt_resultt solve_bound(unsigned k);
  smt_resultt solve_prefix();

protected:
  const transition_systemt &ts;
  const namespacet &ns;
  std::unique_ptr<smt_convt> solver;

  expr2tc bad_literal(unsigned step) const;
  /// Returns false if the prefix itself violates a property. Assumptions
  /// before the loop only hold from here on: each prefix property already
  /// carries the ones that precede it.
  bool check_prefix();
  void assume_prefix();
  void add_step(unsigned step);
  void close_step(unsigned step);
  smt_resultt solve_bad(unsigned step, bool push_pop);
  void report_counterexample(unsigned step);
};
