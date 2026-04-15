/* Regression test: RUP (ROUND_TO_PLUS_INF) interval lifting for ieee_div --
 * numerator tracked, denominator fresh, double precision.
 *
 * PURPOSE
 * -------
 * Verifies the mixed lookup path for RUP ieee_div: when the numerator is
 * tracked in ir_ra_interval_map and the denominator is fresh, the tracked
 * numerator endpoints appear in the hull quotients.
 *
 * PROOF SHAPE (B_dir, RUP, double precision)
 * ------------------------------------------
 * First div:  z = x / y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[z_smt] = {ra_lo_up::0, ra_hi_up::0}
 *
 * Second div:  w = z / x  (z tracked as numerator, x fresh as denominator)
 *   iv(z) = {ra_lo_up::0, ra_hi_up::0}   (from map)
 *   iv(x) = {x_smt, x_smt}               (point fallback)
 *   denom_admissible = (x_smt > 0 || x_smt < 0)
 *   Hull quotients: ra_lo_up::0 / x_smt, ra_hi_up::0 / x_smt
 *   lo_r, hi_r via ITE
 *   ra_lo_up::1 = lo_r               (exact lower)
 *   ra_hi_up::1 = hi_r + B_dir(hi_r) (widened upper)
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_up::0   -- first div's lower bound declared
 *   ra_lo_up::1   -- second div's mixed-path lower bound declared
 *   (/ |smt_conv::ra_lo_up::0|  -- tracked endpoint in hull quotient
 *   (ite           -- ITE for hull sort / admissibility guard
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
  double z = x / y; /* first RUP div: both fresh -> point fallback; stored */
  double w = z / x; /* second RUP div: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z / x exactly. */
  assert(w != z / x);
  return 0;
}
