#ifndef GOTO_PROGRAMS_GOTO_TERMINATION_H_
#define GOTO_PROGRAMS_GOTO_TERMINATION_H_

#include <goto-programs/goto_functions.h>
#include <util/namespace.h>
#include <util/options.h>

/// Reduce non-termination to a reachability safety property.
///
/// Two transformations:
///
///   1. Apply k-induction's per-loop havoc + assume-entry-cond to
///      every function except __ESBMC_main, so the inductive step
///      runs from an arbitrary loop state.
///   2. Insert per-loop ASSERT(false) markers on every edge that
///      leaves a loop (top-test exit, break, early-return-as-goto,
///      goto-out-of-loop, do-while fall-through). Each marker is
///      flagged inductive_step_instruction so base case and forward
///      condition skip it; only the inductive step sees the claim.
///
/// Reduction: the program does NOT terminate iff every marker is
/// unreachable in the inductive step. UNSAT at the marker proves
/// non-termination; SAT in the base case refutes it.
///
/// Soundness gate: if any loop's modified-var set is empty,
/// k-induction's make_nondet_assign inserts no havoc for it
/// (e.g. loops that only write through dereferenced pointers,
/// `*p = ...`). In that case IS would run the loop concretely and a
/// "marker unreachable within k unwindings" UNSAT is no longer a
/// real non-termination witness. The pass sets options'
/// "disable-inductive-step" so the BMC driver treats IS as
/// inconclusive for the program.
void goto_termination(
  goto_functionst &goto_functions,
  optionst &options,
  const namespacet &ns);

#endif /* GOTO_PROGRAMS_GOTO_TERMINATION_H_ */
