/* Regression test: weak (unconstrained) enclosure for long double under --ir-ieee.
 *
 * PURPOSE
 * -------
 * Verifies that the fallback path in apply_ieee754_semantics fires correctly
 * for non-standard FP formats.  long double is 128-bit (fraction=112,
 * exponent=15) on LP64 systems and 96-bit (fraction=80, exponent=15) on
 * ILP32 systems; neither matches the IEEE double (f=52, e=11) or single
 * (f=23, e=8) specs, so it exercises the else-branch:
 *
 *   else {
 *     // theorem-driven bounds not yet implemented for non-standard formats
 *     ra_lo_weak = mk_fresh(Real);
 *     ra_hi_weak = mk_fresh(Real);
 *     assert ra_lo <= result <= ra_hi  and  ra_lo <= ra_hi;
 *   }
 *
 * WHAT IS CHECKED
 * ---------------
 * The test checks that:
 *   (a) ra_lo_weak and ra_hi_weak are still declared (the weak enclosure fires, not a crash)
 *   (b) the run completes with VERIFICATION FAILED
 *
 * Notably, no ITE (absolute-value term) and no eps_rel numerator appear in the
 * formula -- those are hallmarks of the tight-bound path and must be absent
 * here.  The test framework does not support negative assertions, so their
 * absence is not checked directly; the tight-bound regression tests
 * (ra-enclosure-tight-bounds/ and ra-enclosure-tight-bounds-float/) serve as
 * the counterpart positive checks.
 *
 * ASSERTION CHOICE
 * ----------------
 * "z != x + y" is always false in real arithmetic (z was assigned x+y exactly),
 * so the verdict is deterministically VERIFICATION FAILED regardless of how
 * tight the enclosure is.  This makes the verdict a stable run-completion guard.
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   \(declare-fun \|smt_conv::ra_lo_weak::0\| \(\) Real\)  -- weak enclosure declared
 *   \(declare-fun \|smt_conv::ra_hi_weak::0\| \(\) Real\)  -- weak enclosure declared
 *   ^VERIFICATION FAILED$                             -- run completed
 */
#include <assert.h>

extern long double __VERIFIER_nondet_long_double(void);

int main(void)
{
  long double x = __VERIFIER_nondet_long_double();
  long double y = __VERIFIER_nondet_long_double();
  long double z = x + y; /* triggers apply_ieee754_semantics -> ra_lo_weak::0, ra_hi_weak::0 */

  /* Always false in real/integer encoding: z == x+y exactly.
   * Gives a deterministic VERIFICATION FAILED to confirm the run completed. */
  assert(z != x + y);
  return 0;
}
