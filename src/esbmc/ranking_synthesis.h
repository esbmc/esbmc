#ifndef ESBMC_RANKING_SYNTHESIS_H
#define ESBMC_RANKING_SYNTHESIS_H

#include <goto-programs/goto_functions.h>
#include <util/namespace.h>
#include <util/options.h>
#include <util/threeval.h>

/// Linear ranking-function termination check.
///
/// For each loop in the program that matches a supported shape, try a
/// small fixed set of candidate linear ranking functions and discharge
/// two SMT obligations per candidate:
///
///   * decrease:  guard ∧ ¬(m' < m) is UNSAT   (m strictly decreases
///                each iteration), where m' is m with the body's
///                transition applied;
///   * bounded:   guard ∧ (m < L) is UNSAT      (m is bounded below by
///                a guard-implied constant L).
///
/// If every loop has some candidate for which both obligations hold,
/// the program terminates (return TV_TRUE). If any loop cannot be
/// proven (unsupported shape, no candidate works), the check is
/// inconclusive (return TV_UNKNOWN) and the caller falls back to the
/// marker / forward-condition / inductive-step machinery. The check
/// never returns TV_FALSE — it can prove termination but never refute
/// it, so it can only ever turn an UNKNOWN into a SUCCESSFUL, never
/// produce a wrong verdict.
///
/// Must be called BEFORE goto_termination()'s havoc transform, on the
/// original (un-havoced) goto program — the obligations are built from
/// the real loop transition relation.
///
/// The measure is computed in a type wider than its operands so the
/// difference / decrement cannot overflow (signed overflow would be
/// UB and unsound); the lower bound L is derived from the guard, never
/// hardcoded.
tvt try_prove_termination_by_ranking(
  goto_functionst &goto_functions,
  optionst &options,
  const namespacet &ns);

#endif /* ESBMC_RANKING_SYNTHESIS_H */
