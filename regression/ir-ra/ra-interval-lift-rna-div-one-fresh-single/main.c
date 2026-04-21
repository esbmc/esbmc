/* Regression test: RNA (ROUND_TO_AWAY) interval lifting for ieee_div --
 * numerator tracked, denominator fresh, single precision.
 *
 * PURPOSE
 * -------
 * Verifies the single-precision mixed lookup path for RNA ieee_div: tracked
 * numerator endpoints appear in the hull quotients and the single-precision
 * nearest-mode constants are used.
 *
 * PROOF SHAPE (B_near, RNA, single precision)
 * --------------------------------------------
 * First div:  z = x / y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[z_smt] = {ra_lo_aw::0, ra_hi_aw::0}
 *
 * Second div:  w = z / x  (z tracked, x fresh)
 *   iv(z) = {ra_lo_aw::0, ra_hi_aw::0}   (from map)
 *   iv(x) = {x_smt, x_smt}               (point fallback)
 *   Hull quotients contain ra_lo_aw::0 / x_smt, ra_hi_aw::0 / x_smt.
 *   ra_lo_aw::1, ra_hi_aw::1 pinned via single-precision RNA enclosure.
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_aw::0   -- first div's lower bound declared
 *   ra_lo_aw::1   -- second div's mixed-path lower bound declared
 *   (/ |smt_conv::ra_lo_aw::0|  -- tracked endpoint in hull quotient
 *   (ite           -- ITE for hull sort / admissibility guard
 *   5960464477539063  -- Z3 numerator for eps_rel_near = 2^-24 (single)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern float __VERIFIER_nondet_float(void);

int main(void)
{
  __ESBMC_rounding_mode = 1; /* ROUND_TO_AWAY */
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x / y; /* first RNA div: both fresh -> point fallback; stored */
  float w = z / x; /* second RNA div: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z / x exactly. */
  assert(w != z / x);
  return 0;
}
