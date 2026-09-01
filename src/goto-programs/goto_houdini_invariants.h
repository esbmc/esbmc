#ifndef GOTO_PROGRAMS_GOTO_HOUDINI_INVARIANTS_H_
#define GOTO_PROGRAMS_GOTO_HOUDINI_INVARIANTS_H_

#include <goto-programs/goto_functions.h>
#include <optional>
#include <set>
#include <string>

/// Houdini-style loop-invariant inference.
///
/// The affine recogniser in goto_invariant_synthesis.h summarises a loop it can
/// read in closed form. Houdini instead *guesses* a pool of candidate facts and
/// lets the solver delete the ones that are not inductive, so it reaches loops
/// with no counter and no affine update at all -- the shape
///
///   float x = 2; while (nondet_bool()) x = 2*x - 1; assert(x > 0);
///
/// where the property `x > 0` is not itself inductive (a subnormal x makes
/// 2*x - 1 round to -1.0) but the candidate `x > 1` is, and implies it.
///
/// The existing goto_loop_invariant schema already performs Houdini's inner
/// check, which is why this pass is only a candidate generator plus a
/// filter. That schema emits one ASSERT per invariant for the base case and
/// one per invariant for the inductive step, while the havoc step ASSUMEs the
/// whole surviving set. Asserting candidate i under the assumption of every
/// surviving candidate *is* consecution relative to the current set, i.e.
/// exactly the query Houdini iterates on.
///
/// The driver therefore runs a fixpoint: emit the surviving pool, discharge it,
/// delete whichever candidates failed, repeat. Deleting a candidate only
/// weakens the assumed set, so a previously-inductive candidate can fail on a
/// later round; the loop runs to a fixpoint rather than a single pass. It
/// terminates because every round but the last deletes at least one candidate.
///
/// Soundness does not rest on any of this. Candidates are asserted, never
/// assumed-without-proof, so a surviving candidate has had both its base case
/// and its inductive step discharged. A bad guess costs a wasted round, never
/// an unsound proof.

/// Generate the candidate pool and attach one LOOP_INVARIANT instruction per
/// candidate, immediately before each loop head. Returns the ids emitted.
///
/// One instruction per candidate is deliberate: goto_loop_invariant folds the
/// expression *list* of a single instruction into one conjunction and so into
/// one claim, which would make a failure unattributable. Separate instructions
/// keep one claim per candidate.
/// @p keep selects which candidates to emit; nullopt means the pool has not
/// been filtered yet and every candidate is emitted. An *empty* set is not the
/// same thing -- it means every candidate was refuted, so nothing is emitted.
std::set<std::string> goto_houdini_emit_candidates(
  goto_functionst &goto_functions,
  const std::optional<std::set<std::string>> &keep);

#endif /* GOTO_PROGRAMS_GOTO_HOUDINI_INVARIANTS_H_ */
