#pragma once

#include <esbmc/ts/transition_system.h>
#include <solvers/smt/smt_conv.h>

#include <cstdint>
#include <memory>

/// BMC and k-induction over a transition system, unrolled step by step in a
/// single solver. Queries select what they need through activation literals.
class ts_enginet
{
public:
  /// With `bind_init`, the first state is the initial state outright, which
  /// lets the solver fold constant initial values; only BMC can use it.
  ts_enginet(
    const transition_systemt &ts,
    const namespacet &ns,
    optionst &options,
    bool bind_init = false);

  /// Exit codes follow do_bmc_strategy: 0 SUCCESSFUL or UNKNOWN, 1 FAILED,
  /// 6 solver error.
  int bmc(uint64_t max_k, bool push_pop);
  int k_induction(uint64_t max_k, bool simple_path);

protected:
  const transition_systemt &ts;
  const namespacet &ns;
  std::unique_ptr<smt_convt> solver;
  expr2tc act_init, act_inv;

  expr2tc bad_literal(unsigned step) const;
  /// Returns false if the prefix itself violates a property.
  bool check_prefix();
  void add_step(unsigned step);
  void close_step(unsigned step);
  smt_resultt solve_bad(unsigned step, bool from_init, bool push_pop);
  void report_counterexample(unsigned step);
  unsigned add_simple_path_lemmas(unsigned step);
};
