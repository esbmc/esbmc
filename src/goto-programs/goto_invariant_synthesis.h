#ifndef GOTO_PROGRAMS_GOTO_INVARIANT_SYNTHESIS_H_
#define GOTO_PROGRAMS_GOTO_INVARIANT_SYNTHESIS_H_

#include <goto-programs/goto_functions.h>
#include <irep2/irep2_expr.h>

/// What goto_check will instrument on the guards this pass emits. It checks
/// every instruction guard, including the synthesised ones, so a closed form
/// emitted at a type it watches draws overflow claims on arithmetic the user
/// never wrote.
struct overflow_checkst
{
  /// --overflow-check: signed arithmetic is instrumented.
  bool signed_arith = false;
  /// --unsigned-overflow-check: unsigned arithmetic is instrumented too
  /// (goto_check.cpp, `enable_unsigned_overflow_check`). Under it no integer
  /// type is safe to emit the closed form at, so synthesis declines outright.
  bool unsigned_arith = false;
};

/// The recogniser's pure syntactic predicates. Exposed for unit testing: every
/// other route to them runs through a frontend and a solver, where a predicate
/// that never fires is indistinguishable from one that answers correctly.
namespace invariant_synthesis
{
/// Split `cond` into counter and bound for the `<`/`<=` shapes this pass
/// handles, and report which one it was. Other comparisons (and decrementing
/// loops) are left to a later revision.
bool split_bound(
  const expr2tc &cond,
  expr2tc &counter,
  expr2tc &bound,
  bool &inclusive);

/// `lhs = lhs + addend` -- the only body assignment shape recognised here.
bool is_self_increment(
  const expr2tc &target,
  const expr2tc &source,
  expr2tc &addend);

/// Whether the two-disjunct bound `(i <op> B) || i == E` is established from a
/// counter entry value of `entry`. See the definition for the case analysis.
bool entry_admits_two_disjunct_bound(const expr2tc &entry, bool inclusive);
} // namespace invariant_synthesis

/// Synthesise loop invariants for affine counter/accumulator loops and attach
/// them as LOOP_INVARIANT instructions, exactly as if the user had written
/// __ESBMC_loop_invariant(). The subsequent goto_loop_invariant pass discharges
/// them through its assert/havoc-assume/assert schema, so a wrong candidate
/// fails an assertion rather than being assumed: synthesis cannot make an
/// unsound proof, only a spurious failure or a useless invariant.
///
/// Recognised shape, for a loop whose head is `IF !(i <op> B) GOTO exit` with
/// straight-line body:
///
///   i = i + 1                  counter, unit step
///   s = s + e                  accumulator, e free of loop-modified vars
///
/// yields, with i0/s0 the entry values and E the exit value of i:
///
///   s == s0 + (i - i0) * e                 accumulator closed form
///   (i <op> B) || i == E                   counter bound
///   (+ i == i0)                            third arm, constant-addend regime
///   (+ i0 <op> B || i == i0)               never-entered, constant-addend
///   (+ i >= i0)                            unsigned counters only
///
/// THE ONE DESIGN CONSTRAINT, from which every restriction below follows.
///
/// The bound is a disjunction rather than the tighter `i <= B + 1` so that
/// negating the guard at the exit yields the *equality* i == E by disjunct
/// elimination. Substituting that equality lets the two `* e` terms share a
/// multiplier. The inequality form instead leaves the solver proving two
/// 64-bit multiplier circuits equivalent, which does not terminate — and every
/// additional live arm at the exit costs the same way. Measured on the
/// accumulator loop above: two disjuncts discharge in ~1s, three do not finish
/// in 120s.
///
/// That cost exists only when an addend is *symbolic*. With every addend a
/// literal there is no multiplier to miter and extra arms are free: measured on
/// the sum01 exit obligation, three disjuncts with a constant addend discharge
/// in 0s against 45s-and-counting for the same shape with a symbolic one.
/// Hence two regimes:
///
///   symbolic addend   two disjuncts only. Establishment then needs an
///                     unsigned counter entering at 0 or 1 — see
///                     entry_admits_two_disjunct_bound for the case analysis.
///
///   literal addends   the third arm `i == i0` is affordable, which makes
///                     establishment unconditional, which in turn admits
///                     signed counters and any literal entry value. Also
///                     carries the never-entered arm, without which the exit
///                     admits i == E for a bound that never satisfied the
///                     guard and the closed form reports an accumulator the
///                     loop could not produce.
///
/// TWO PRECONDITIONS THE ABSTRACTION ITSELF DOES NOT ENFORCE.
///
/// Concurrency. Cutting a loop deletes its interleaving points, so a claim
/// another thread could only violate through one of them is no longer reachable
/// in the cut program. What keeps that from being a false proof is not the
/// abstraction: it is #7491's classifier, which reports every claim downstream
/// of a havoc as UNKNOWN rather than SUCCESSFUL
/// (regression/esbmc/synth_loop_invariant_thread pins this). Were that
/// classifier ever wrong for a cross-thread claim, this shape becomes unsound.
/// It is a dependency of the design, not a property of it.
///
/// A user-written invariant is authoritative. A synthesised marker on the same
/// loop is declined (has_user_invariant), and so is synthesis inside any
/// function a user invariant's expression calls, transitively -- otherwise the
/// user's own marker reads a havoc-abstracted return value. See
/// collect_invariant_dependencies. It is a call-graph rule and nothing wider:
/// the same value reaching a marker through a local this pass has cut is not
/// declined, and does not need to be -- the closed form describes that local
/// exactly, where a cut callee's return value is only havoc plus whatever the
/// invariant on its own loop happens to say.
///
/// `i >= i0` prunes havoced states below the entry value, where `i - i0` wraps.
/// It is emitted for unsigned counters only: a signed `i == n == INT_MAX` still
/// satisfies the guard, so the body's `i + 1` wraps and the conjunct is false
/// after a legitimate iteration. Signed loops therefore carry a weaker bound
/// and are declined where that weakness is observable — a body that asserts, or
/// a run with signed overflow checking on. --unsigned-overflow-check
/// declines every loop, signed or not; see overflow_checkst.
/// `k_induction_ran` reports whether goto_k_induction has already rewritten the
/// loop heads. It is a diagnostic input only: the recogniser matches on a head
/// this pass then no longer finds, so the run is a no-op and the user is told
/// why rather than left with a silent one.
void goto_synthesise_loop_invariants(
  goto_functionst &goto_functions,
  const overflow_checkst &overflow,
  bool k_induction_ran);

#endif /* GOTO_PROGRAMS_GOTO_INVARIANT_SYNTHESIS_H_ */
