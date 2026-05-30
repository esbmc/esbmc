#ifndef ESBMC_NON_TERMINATION_H
#define ESBMC_NON_TERMINATION_H

#include <goto-programs/goto_functions.h>
#include <util/namespace.h>
#include <util/options.h>
#include <util/threeval.h>

/// Recurrent-set non-termination check (Gupta-Henzinger-Majumdar-
/// Rybalchenko-Xu, POPL 2008).
///
/// For each `while(1)`-shaped loop in the program, look for an
/// inductive recurrent set R: a state predicate such that
///   (1) the initial state satisfies R,
///   (2) R is closed under one body iteration: there exists an input
///       choice such that body(R, input) ⊆ R, and
///   (3) R is disjoint from any exit path the body can take (calls to
///       exit/abort/__VERIFIER_error, returns from main, etc.).
///
/// If such an R exists, the loop has an infinite execution (the demon
/// always picks the input that keeps state in R), so the program is
/// non-terminating.
///
/// Returns:
///   * TV_FALSE — proved non-terminating (caller reports VERIFICATION
///                FAILED for the termination property).
///   * TV_UNKNOWN — could not prove non-termination (caller falls back
///                  to the existing forward-condition / inductive-step
///                  machinery).
///
/// Never returns TV_TRUE — this checker only refutes termination, it
/// does not certify it. The corresponding certifier lives in
/// ranking_synthesis.cpp.
///
/// MVP scope (this commit): only the eca-rers2012 finite-state-machine
/// shape — a `while(1)` whose body reads an input, validates it, calls
/// a single update function, and back-edges. The recurrent-set
/// predicate is restricted to a conjunction of equalities over global
/// state variables, seeded from their constant initialisations.
tvt try_prove_non_termination_by_recurrent_set(
  goto_functionst &goto_functions,
  optionst &options,
  const namespacet &ns);

#endif /* ESBMC_NON_TERMINATION_H */
