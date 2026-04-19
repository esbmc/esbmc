/* Regression test: RUP (ROUND_TO_PLUS_INF) interval lifting for ieee_mul --
 * one tracked, one fresh, double precision.
 *
 * PURPOSE
 * -------
 * Verifies the mixed lookup path for RUP ieee_mul: when one operand of a
 * second RUP ieee_mul is tracked in ir_ra_interval_map and the other is a
 * fresh nondet variable, the tracked operand uses its stored interval while
 * the fresh one falls back to the point interval {side, side}.
 *
 * PROOF SHAPE (B_dir, RUP, double precision)
 * ------------------------------------------
 * First mul:  z = x * y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[real_z] = {ra_lo_up::0, ra_hi_up::0}
 *
 * Second mul:  w = z * x  (z tracked, x fresh -> mixed path)
 *   iv(z) = {ra_lo_up::0, ra_hi_up::0}   (from map)
 *   iv(x) = {x_smt, x_smt}               (point fallback)
 *   p1 = ra_lo_up::0 * x_smt
 *   p3 = ra_hi_up::0 * x_smt
 *   lo_r = min(p1,p3),  hi_r = max(p1,p3)
 *   ra_lo_up::1 = lo_r          (exact lower)
 *   ra_hi_up::1 = hi_r + B_dir(hi_r)
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_up::0   -- first mul's interval lower bound declared
 *   ra_lo_up::1   -- second mul's mixed-path lower bound declared
 *   (* |smt_conv::ra_lo_up::0|  -- tracked endpoint in hull product
 *   (ite           -- nested ITE for min/max hull and |r| in B_dir
 *   22204460492503131  -- Z3 numerator for eps_rel_dir = 2^-52 (double)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 2; /* ROUND_TO_PLUS_INF */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x * y; /* first RUP mul: both fresh -> point fallback; stored */
  double w = z * x; /* second RUP mul: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z * x exactly. */
  assert(w != z * x);
  return 0;
}
