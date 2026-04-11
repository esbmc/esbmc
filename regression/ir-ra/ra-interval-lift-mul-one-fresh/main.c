/* Regression test: RNE interval lifting for ieee_mul -- one tracked, one fresh.
 *
 * PURPOSE
 * -------
 * Verifies the mixed lookup path for ieee_mul: when one operand of a second
 * RNE ieee_mul is tracked in ir_ra_interval_map and the other is a fresh
 * nondet variable, the tracked operand uses its stored interval while the
 * fresh one falls back to the point interval {side, side}.
 *
 * PROOF SHAPE (B_near, RNE, double precision)
 * -------------------------------------------
 * First mul:  z = x * y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[real_z] = {ra_lo::0, ra_hi::0}
 *
 * Second mul:  w = z * x  (z tracked, x fresh -> mixed path)
 *   iv(z) = {ra_lo::0, ra_hi::0}   (from map)
 *   iv(x) = {x_smt, x_smt}         (point fallback)
 *   p1 = ra_lo::0 * x_smt
 *   p2 = ra_lo::0 * x_smt  (= p1, since iv(x) is a point)
 *   p3 = ra_hi::0 * x_smt
 *   p4 = ra_hi::0 * x_smt  (= p3)
 *   lo_r = min(p1,p3) = ITE(ra_lo::0 * x_smt <= ra_hi::0 * x_smt, ...)
 *   hi_r = max(p1,p3)
 *   ra_lo::1 = lo_r - B_near(lo_r)
 *   ra_hi::1 = hi_r + B_near(hi_r)
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo::0   -- first mul's interval lower bound declared
 *   ra_lo::1   -- second mul's mixed-path lower bound declared
 *   (* |smt_conv::ra_lo::0|  -- tracked endpoint in hull product
 *   (ite        -- nested ITE for min/max hull and absolute value
 *   5551115123125783  -- Z3 numerator for eps_rel = 2^-53 (double)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x * y; /* first RNE mul: both fresh -> point fallback; stored */
  double w = z * x; /* second RNE mul: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z * x exactly. */
  assert(w != z * x);
  return 0;
}
