#ifndef GOTO_PROGRAMS_GOTO_INVARIANT_SYNTHESIS_H_
#define GOTO_PROGRAMS_GOTO_INVARIANT_SYNTHESIS_H_

#include <goto-programs/goto_functions.h>

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
/// yields, with i0/s0 the entry values and E the exit value of i,
///
///   (i <op> B) || i == E                   counter bound
///   s == s0 + (i - i0) * e                 accumulator closed form
///
/// The bound is emitted as a disjunction rather than the tighter `i <= B + 1`
/// so that negating the guard at the loop exit yields the *equality* i == E by
/// disjunct elimination. Substituting that equality lets the two `* e` terms
/// share a multiplier; the inequality form instead leaves the solver proving
/// two 64-bit multiplier circuits equivalent, which does not terminate.
///
/// For the same reason the bound stays at exactly two disjuncts. A third arm
/// covering a loop that never runs (`i == i0`) is what a general entry value
/// would need, but it leaves two multiplier branches alive at the exit and the
/// query stops terminating; measured on the accumulator loop above, two
/// disjuncts discharge in ~1s where three do not finish in 120s. Synthesis is
/// therefore restricted to the entry values for which two disjuncts are already
/// establishable — see entry_admits_two_disjunct_bound.
void goto_synthesise_loop_invariants(goto_functionst &goto_functions);

#endif /* GOTO_PROGRAMS_GOTO_INVARIANT_SYNTHESIS_H_ */
