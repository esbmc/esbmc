/* Regression test: RNE interval lifting for ieee_mul -- one tracked, one fresh,
 * single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-mul-one-fresh but exercises the single-precision
 * (float) path. Verifies the mixed lookup path: when one operand of a second
 * RNE ieee_mul is tracked and the other is a fresh nondet variable.
 *
 * PROOF SHAPE (B_near, RNE, single precision)
 * -------------------------------------------
 * Second mul:  w = z * x  (z tracked, x fresh -> mixed path)
 *   lo_r = min(ra_lo::0 * x_smt, ra_hi::0 * x_smt)
 *   hi_r = max(ra_lo::0 * x_smt, ra_hi::0 * x_smt)
 *   RTZ three-way ITE on sign not present (RNE uses symmetric B_near)
 *   ra_lo::1 and ra_hi::1 pinned accordingly
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo::0   -- first mul's interval lower bound declared
 *   ra_lo::1   -- second mul's mixed-path lower bound declared
 *   (* |smt_conv::ra_lo::0|  -- tracked endpoint in hull product
 *   (ite        -- nested ITE present
 *   5960464477539063  -- Z3 numerator for eps_rel = 2^-24 (single)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern float __VERIFIER_nondet_float(void);

int main(void)
{
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x * y; /* first RNE mul: both fresh -> point fallback; stored */
  float w = z * x; /* second RNE mul: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z * x exactly. */
  assert(w != z * x);
  return 0;
}
