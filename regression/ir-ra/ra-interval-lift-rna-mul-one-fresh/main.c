/* Regression test: RNA (ROUND_TO_AWAY) interval lifting for ieee_mul --
 * one tracked, one fresh.
 *
 * PURPOSE
 * -------
 * Verifies the mixed lookup path for RNA ieee_mul: when one operand of a
 * second RNA ieee_mul is tracked in ir_ra_interval_map and the other is a
 * fresh nondet variable, the tracked operand uses its stored interval while
 * the fresh one falls back to the point interval {side, side}.
 *
 * PROOF SHAPE (B_near, RNA, double precision)
 * -------------------------------------------
 * First mul:  z = x * y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[real_z] = {ra_lo_aw::0, ra_hi_aw::0}
 *
 * Second mul:  w = z * x  (z tracked, x fresh -> mixed path)
 *   iv(z) = {ra_lo_aw::0, ra_hi_aw::0}   (from map)
 *   iv(x) = {x_smt, x_smt}               (point fallback)
 *   p1 = ra_lo_aw::0 * x_smt
 *   p2 = ra_lo_aw::0 * x_smt  (= p1, iv(x) is a point)
 *   p3 = ra_hi_aw::0 * x_smt
 *   p4 = ra_hi_aw::0 * x_smt  (= p3)
 *   lo_r = min(p1,p3),  hi_r = max(p1,p3)  via ITE on sign of x
 *   ra_lo_aw::1 = lo_r - B_near(lo_r)
 *   ra_hi_aw::1 = hi_r + B_near(hi_r)
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_aw::0   -- first mul's interval lower bound declared
 *   ra_lo_aw::1   -- second mul's mixed-path lower bound declared
 *   (* |smt_conv::ra_lo_aw::0|  -- tracked endpoint in hull product
 *   (ite           -- nested ITE for min/max hull and absolute value
 *   5551115123125783  -- Z3 numerator for eps_rel = 2^-53 (double, same as RNE)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 1; /* ROUND_TO_AWAY */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x * y; /* first RNA mul: both fresh -> point fallback; stored */
  double w = z * x; /* second RNA mul: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z * x exactly. */
  assert(w != z * x);
  return 0;
}
