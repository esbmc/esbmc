/* Regression test: RNE (ROUND_TO_EVEN) interval lifting for ieee_div --
 * numerator tracked, denominator fresh, double precision.
 *
 * PURPOSE
 * -------
 * Verifies the mixed lookup path for RNE ieee_div: when the numerator of a
 * second RNE ieee_div is tracked in ir_ra_interval_map and the denominator
 * is a fresh nondet variable, the tracked numerator uses its stored interval
 * while the denominator is used as a point value (conservative design to
 * avoid zero-crossing unsoundness in the division hull).
 *
 * PROOF SHAPE (B_near, RNE, double precision)
 * -------------------------------------------
 * First div:  z = x / y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[z_smt] = {ra_lo::0, ra_hi::0}
 *
 * Second div:  w = z / x  (z tracked as numerator, x fresh as denominator)
 *   iv(z) = {ra_lo::0, ra_hi::0}   (from map)
 *   denominator = x_smt             (point -- never lifted)
 *   d_lo = ra_lo::0 / x_smt
 *   d_hi = ra_hi::0 / x_smt
 *   lo_r = min(d_lo, d_hi) via ITE on sign of x
 *   hi_r = max(d_lo, d_hi) via ITE on sign of x
 *   ra_lo::1, ra_hi::1 pinned via RNE enclosure
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo::0   -- first div's lower bound declared
 *   ra_lo::1   -- second div's mixed-path lower bound declared
 *   (/ |smt_conv::ra_lo::0|  -- tracked endpoint in hull quotient
 *   (ite           -- ITE for hull min/max
 *   5551115123125783  -- Z3 numerator for eps_rel_near = 2^-53 (double)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x / y; /* first RNE div: both fresh -> point fallback; stored */
  double w = z / x; /* second RNE div: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z / x exactly. */
  assert(w != z / x);
  return 0;
}
