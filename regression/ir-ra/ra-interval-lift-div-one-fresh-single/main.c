/* Regression test: RNE (ROUND_TO_EVEN) interval lifting for ieee_div --
 * numerator tracked, denominator fresh, single precision.
 *
 * PURPOSE
 * -------
 * Verifies the single-precision mixed lookup path for RNE ieee_div: when the
 * numerator is tracked and the denominator is fresh, the tracked interval
 * endpoints appear in the hull quotients and the single-precision RNE
 * enclosure constants are used.
 *
 * PROOF SHAPE (B_near, RNE, single precision)
 * --------------------------------------------
 * First div:  z = x / y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[z_smt] = {ra_lo::0, ra_hi::0}
 *
 * Second div:  w = z / x  (z tracked as numerator, x fresh as denominator)
 *   iv(z) = {ra_lo::0, ra_hi::0}   (from map)
 *   d_lo = ra_lo::0 / x_smt
 *   d_hi = ra_hi::0 / x_smt
 *   lo_r, hi_r via ITE
 *   ra_lo::1, ra_hi::1 pinned via single-precision RNE enclosure
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo::0   -- first div's lower bound declared
 *   ra_lo::1   -- second div's mixed-path lower bound declared
 *   (/ |smt_conv::ra_lo::0|  -- tracked endpoint in hull quotient
 *   (ite           -- ITE for hull min/max
 *   5960464477539063  -- Z3 numerator for eps_rel_near = 2^-24 (single)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern float __VERIFIER_nondet_float(void);

int main(void)
{
  __ESBMC_rounding_mode = 0; /* ROUND_TO_EVEN */
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x / y; /* first RNE div: both fresh -> point fallback; stored */
  float w = z / x; /* second RNE div: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z / x exactly. */
  assert(w != z / x);
  return 0;
}
