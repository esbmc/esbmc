#ifndef ESBMC_NON_TERMINATION_H
#define ESBMC_NON_TERMINATION_H

#include <goto-programs/goto_functions.h>
#include <util/symtab/namespace.h>
#include <util/config/options.h>
#include <util/base/threeval.h>

/// Period-1 fixpoint non-termination check, restricted to the eca-
/// rers2012 main-loop shape.
///
/// Recognises a `while(1)` whose body is exactly
///     input = NONDET; if (input not in {v1, ..., vn}) return;
///     output = calculate_output(input);
/// and asks SMT whether there is a state s and input i in {v1, ..., vn}
/// such that NONE of the IF branches in `calculate_output` fire. Each
/// branch in the callee transfers control out of the body (either by
/// RETURN or by a call to exit/abort), so "no branch fires" implies the
/// fall-through tail (`return -2`) runs without touching any state.
/// If that question is SAT, the demon can pick (s, i) forever, the
/// loop never terminates, and we return TV_FALSE.
///
/// Returns:
///   * TV_FALSE — period-1 fixpoint exists; caller reports VERIFICATION
///                FAILED for the termination property.
///   * TV_UNKNOWN — no eca-shape loop matched, or the SMT discharge
///                  was UNSAT/UNKNOWN; caller falls back to the
///                  ranking-function pass / k-induction.
///
/// Never returns TV_TRUE. This is a refutation-only checker. On TV_FALSE,
/// @p loop_location is the head of the non-terminating loop.
///
/// The check is sound because the SMT obligation directly witnesses a
/// non-terminating execution: the witness state and input cause the
/// callee to fall through with no writes, so the next iteration is
/// identical, ad infinitum. A previous syntactic prototype (find a
/// clean non-exiting path in the goto-program) lost to mislabelled
/// LDV benchmarks at SV-COMP's -16/+1 weighting; the period-1 fixpoint
/// check sidesteps that by only flagging loops where the SMT witness
/// is unambiguous.
tvt try_prove_non_termination_by_recurrent_set(
  goto_functionst &goto_functions,
  optionst &options,
  const namespacet &ns,
  locationt &loop_location);

/// Before k-induction transforms the program: for each input loop of the
/// entry function whose iterations are a function of its globals and a finite
/// input, enumerate the global states it reaches at its head from the initial
/// state (at most @p max_states) and assume them there in the inductive step.
/// Returns each such loop's invariant, as a C expression, at its loop head.
std::vector<std::pair<locationt, std::string>> assume_reachable_loop_states(
  goto_functionst &goto_functions,
  const namespacet &ns,
  size_t max_states);

#endif /* ESBMC_NON_TERMINATION_H */
