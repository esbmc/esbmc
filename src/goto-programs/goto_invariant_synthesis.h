#ifndef GOTO_PROGRAMS_GOTO_INVARIANT_SYNTHESIS_H_
#define GOTO_PROGRAMS_GOTO_INVARIANT_SYNTHESIS_H_

#include <goto-programs/goto_functions.h>

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
/// `i >= i0` prunes havoced states below the entry value, where `i - i0` wraps.
/// It is emitted for unsigned counters only: a signed `i == n == INT_MAX` still
/// satisfies the guard, so the body's `i + 1` wraps and the conjunct is false
/// after a legitimate iteration. Signed loops therefore carry a weaker bound
/// and are declined where that weakness is observable — a body that asserts, or
/// a run with signed overflow checking on. --unsigned-overflow-check
/// declines every loop, signed or not; see overflow_checkst.
void goto_synthesise_loop_invariants(
  goto_functionst &goto_functions,
  const overflow_checkst &overflow);

#endif /* GOTO_PROGRAMS_GOTO_INVARIANT_SYNTHESIS_H_ */
