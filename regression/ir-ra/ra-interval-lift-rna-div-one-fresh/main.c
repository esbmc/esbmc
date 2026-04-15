/* Regression test: RNA (ROUND_TO_AWAY) interval lifting for ieee_div --
 * numerator tracked, denominator fresh, double precision.
 *
 * PURPOSE
 * -------
 * Verifies the mixed lookup path for RNA ieee_div: when the numerator is
 * tracked in ir_ra_interval_map and the denominator is a fresh nondet
 * variable, the tracked numerator interval contributes to the hull quotients.
 *
 * PROOF SHAPE (B_near, RNA, double precision)
 * -------------------------------------------
 * First div:  z = x / y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[z_smt] = {ra_lo_aw::0, ra_hi_aw::0}
 *
 * Second div:  w = z / x  (z tracked as numerator, x fresh as denominator)
 *   iv(z) = {ra_lo_aw::0, ra_hi_aw::0}   (from map)
 *   iv(x) = {x_smt, x_smt}               (point fallback)
 *   denom_admissible = (x_smt > 0 || x_smt < 0)
 *   Full hull q1=q2=ra_lo_aw::0/x_smt, q3=q4=ra_hi_aw::0/x_smt
 *   lo_r, hi_r via ITE; ra_lo_aw::1, ra_hi_aw::1 pinned via RNA enclosure
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_aw::0   -- first div's lower bound declared
 *   ra_lo_aw::1   -- second div's mixed-path lower bound declared
 *   (/ |smt_conv::ra_lo_aw::0|  -- tracked endpoint in hull quotient
 *   (ite           -- ITE for hull sort / admissibility guard
 *   5551115123125783  -- Z3 numerator for eps_rel_near = 2^-53 (double)
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
  double z = x / y; /* first RNA div: both fresh -> point fallback; stored */
  double w = z / x; /* second RNA div: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z / x exactly. */
  assert(w != z / x);
  return 0;
}
