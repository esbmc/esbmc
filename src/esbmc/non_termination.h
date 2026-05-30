#ifndef ESBMC_NON_TERMINATION_H
#define ESBMC_NON_TERMINATION_H

#include <goto-programs/goto_functions.h>
#include <util/namespace.h>
#include <util/options.h>
#include <util/threeval.h>

/// Non-termination check entry point.
///
/// Currently a no-op stub that always returns TV_UNKNOWN. The slot is
/// wired into the termination driver in esbmc_parseoptions.cpp so a
/// future recurrent-set / lasso detector can drop in without changing
/// the driver.
///
/// A previous prototype (trivial-recurrent-set detector at the GOTO
/// level — find a non-exiting path through a `while(1)` body that
/// writes no live-across-iterations symbol) was measured on the
/// SV-COMP termination set and net-lost: zero hits on eca-rers2012
/// (every update path modifies state) and net -188 raw on the
/// ldv-linux-3.4-simple "infinite_withcheck_stateful" cluster, because
/// structurally-identical loops in that cluster carry opposite
/// expected_verdict labels. See the implementation file for the full
/// post-mortem.
///
/// Contract:
///   * TV_FALSE — proved non-terminating (caller reports VERIFICATION
///                FAILED for the termination property).
///   * TV_UNKNOWN — could not prove non-termination; the verdict is
///                  delegated to the ranking-function pass and
///                  k-induction.
///
/// Never returns TV_TRUE — this slot is for refutation only. The
/// corresponding certifier lives in ranking_synthesis.cpp.
tvt try_prove_non_termination_by_recurrent_set(
  goto_functionst &goto_functions,
  optionst &options,
  const namespacet &ns);

#endif /* ESBMC_NON_TERMINATION_H */
