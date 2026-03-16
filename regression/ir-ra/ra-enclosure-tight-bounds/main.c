/* Regression test: theorem-driven tight enclosure under --ir-ra.
 *
 * PURPOSE
 * -------
 * Verifies that the SMT formula emitted by --ir-ra for a double-precision
 * addition contains the structural hallmarks of the NEW theorem-driven
 * encoding, NOT just the old containment-only (weak) encoding.
 *
 * OLD weak encoding (commit b83680d40e):
 *   (declare-fun |smt_conv::ra_lo::N| () Real)
 *   (declare-fun |smt_conv::ra_hi::N| () Real)
 *   (assert (<= |ra_lo| result))        <- containment only
 *   (assert (<= result  |ra_hi|))
 *   (assert (<= |ra_lo| |ra_hi|))
 *   No ITE.  No epsilon constants.
 *
 * NEW theorem-driven encoding:
 *   Same declarations, PLUS four pinning constraints that tie ra_lo/ra_hi
 *   to the sound symmetric error enclosure:
 *     B(r)  = eps_rel * |r| + eps_abs
 *     ra_lo = r - B(r),   ra_hi = r + B(r)
 *   encoded as:
 *     (ite (< r 0.0) (- 0.0 r) r)               <- |r| via ITE
 *     (* (/ 2220446049250313.0 ...) |r|)         <- eps_rel = 2^-53
 *
 * WHY THE VERDICT ALONE IS NOT ENOUGH
 * ------------------------------------
 * In integer/real encoding the C variable z is assigned the *exact* real sum
 * x+y.  ra_lo and ra_hi are SMT-internal auxiliary symbols that do not feed
 * back into z.  Both encodings therefore produce the same verification
 * verdict for any C-level assertion.  Only the formula content differs.
 *
 * ASSERTION CHOICE
 * ----------------
 * "z != x + y" is always false in real arithmetic (z was just assigned x+y),
 * so the verdict is deterministically VERIFICATION FAILED regardless of
 * solver search.  This makes the verdict check a stable sanity-of-run guard
 * rather than an encoding-sensitive signal.
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   \(declare-fun \|smt_conv::ra_lo::0\| \(\) Real\)  -- symbol declared
 *   \(declare-fun \|smt_conv::ra_hi::0\| \(\) Real\)  -- symbol declared
 *   \(ite                                              -- |r| absolute value
 *   2220446049250313                                   -- eps_rel numerator
 *   ^VERIFICATION FAILED$                             -- run completed
 *
 * The first four patterns are absent in the old weak encoding; the last is
 * present in both.  A revert to the weak encoding breaks the first four.
 */
#include <assert.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x + y; /* triggers apply_ieee754_semantics -> ra_lo::0, ra_hi::0 */

  /* Always false in real/integer encoding: z == x+y exactly.
   * Gives a deterministic VERIFICATION FAILED to confirm the run completed. */
  assert(z != x + y);
  return 0;
}
