/* Regression test: theorem-driven tight enclosure under --ir-ra, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-enclosure-tight-bounds/ but exercises the IEEE single-precision
 * (float) path instead of double.  Verifies that the SMT formula emitted by
 * --ir-ra for a float addition contains the structural hallmarks of the NEW
 * theorem-driven encoding, not just the old containment-only (weak) encoding.
 *
 * OLD weak encoding:
 *   (declare-fun |smt_conv::ra_lo::N| () Real)
 *   (declare-fun |smt_conv::ra_hi::N| () Real)
 *   (assert (<= |ra_lo| result))
 *   (assert (<= result  |ra_hi|))
 *   (assert (<= |ra_lo| |ra_hi|))
 *   No ITE.  No epsilon constants.
 *
 * NEW theorem-driven encoding also emits:
 *   B(r)  = eps_rel * |r| + eps_abs
 *   ra_lo = r - B(r),   ra_hi = r + B(r)
 * where for single precision (IEEE 754 binary32):
 *   eps_rel = 2^-24 ~= 5.960464477539063e-08
 *           = (/ 5960464477539063.0 100000000000000000000000.0)  in SMT-LIB
 *   eps_abs = 2^-149 (minimum positive subnormal)
 * and |r| is encoded as:
 *   (ite (< r 0.0) (- 0.0 r) r)
 *
 * WHY THE VERDICT ALONE IS NOT ENOUGH
 * ------------------------------------
 * In integer/real encoding the C variable z is the exact real sum x+y.
 * ra_lo and ra_hi are SMT-internal auxiliary symbols that do not constrain z.
 * Both encodings produce the same verdict for any C-level assertion; only the
 * formula content differs.
 *
 * ASSERTION CHOICE
 * ----------------
 * "z != x + y" is always false in real arithmetic (z was assigned x+y), so
 * VERIFICATION FAILED is deterministic and independent of encoding strength.
 * It serves as a run-completion guard, not an encoding-sensitive signal.
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   \(declare-fun \|smt_conv::ra_lo::0\| \(\) Real\)  -- symbol declared
 *   \(declare-fun \|smt_conv::ra_hi::0\| \(\) Real\)  -- symbol declared
 *   \(ite                                              -- |r| absolute value
 *   5960464477539063                                   -- eps_rel numerator (2^-24)
 *   ^VERIFICATION FAILED$                             -- run completed
 *
 * The first four patterns are absent in the old weak encoding.  The eps_rel
 * numerator 5960464477539063 is also distinct from the double-precision
 * numerator 2220446049250313, so the test specifically targets the
 * single-precision path in apply_ieee754_semantics.
 */
#include <assert.h>

extern float __VERIFIER_nondet_float(void);

int main(void)
{
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x + y; /* triggers apply_ieee754_semantics -> ra_lo::0, ra_hi::0 */

  /* Always false in real/integer encoding: z == x+y exactly.
   * Gives a deterministic VERIFICATION FAILED to confirm the run completed. */
  assert(z != x + y);
  return 0;
}
