/// Algorithms to be applied for incremental approaches.

#pragma once

#include <util/algorithms.h>
#include <memory>
#include <assert.h>

/**
 * Translates SSA steps into SMT formulas incrementally.
 * Supports two modes:
 *
 * 1. Offline mode (via run/solve): converts all steps at once, then checks via solve(). 
 *    Used mostly for testing as the regular flow deals with more cases.
 * 2. Online mode (init + step_online): processes steps one at a time. Each
 *    assignment/assumption is added to the solver context immediately.
 *
 * Each assertion is validated via push/pop, returning a TVT. `ask_solver_question` may be
 * called at any point between steps and does not impact the context.
 */
class incremental_smt_algorithm : public ssa_step_algorithm
{
public:
  incremental_smt_algorithm(std::unique_ptr<smt_convt> _ptr, bool sideffect)
    : ssa_step_algorithm(sideffect),
      conv(std::move(_ptr)),
      assumpt_ast(nullptr),
      ignored_count(0),
      output_count(0)
  {
  }

  // Offline mode  
  bool run(SSA_stepst &steps) override;
    /// How many steps were ignored after this algorithm
  BigInt ignored() const override;
  smt_convt::resultt solve(); // for batch mode

  /// Reset solver state. Must be called before the first step_online() call.
  void init();

  /**
   * Process a single SSA step and update the persistent solver context.
   *
   * For assignment/renumber/output/branching/skip steps the step is
   * encoded into the solver and TV_UNKNOWN is returned.
   *
   * For assert/assume steps the encoded condition is checked right away.
   * The returned TVT indicates whether the assertion holds (TV_TRUE), 
   *is violated (TV_FALSE), or is undetermined (TV_UNKNOWN).
   */
  tvt step_online(SSA_stept &step);
  
  void run_on_assignment(SSA_stept &) override;
  void run_on_assume(SSA_stept &) override;
  void run_on_assert(SSA_stept &) override;

  void run_on_output(SSA_stept &) override;
  void run_on_renumber(SSA_stept &) override;
  void run_on_branching(SSA_stept &) override;

  /**
   * Ask whether a boolean expression is always true, always false, or
   * undetermined under the current accumulated assumptions and assignments.
   * Uses scoped push/pop and does not alter the base solver context.
   */
  tvt ask_solver_question(const expr2tc &question);

private:
  std::unique_ptr<smt_convt> conv;
  smt_astt assumpt_ast;
  smt_convt::ast_vec assertions;
  BigInt ignored_count;
  unsigned output_count;
};
