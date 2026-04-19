/* Regression test: RNE (ROUND_TO_EVEN) interval lifting for ieee_div --
 * both operands tracked, double precision.
 *
 * PURPOSE
 * -------
 * Verifies proof-aligned compositional interval lifting for RNE ieee_div
 * when both the numerator and denominator were themselves results of a prior
 * tracked RNE ieee_div. The implementation uses the full four-endpoint hull
 * formula when the denominator interval is admissible (does not contain zero),
 * and falls back to point denominator otherwise.
 *
 * PROOF SHAPE (B_near, RNE, double precision)
 * -------------------------------------------
 * First div:  z = x / y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[z_smt] = {ra_lo::0, ra_hi::0}
 *
 * Second div:  w = z / z  (both operands tracked)
 *   iv(z) = {ra_lo::0, ra_hi::0} for numerator AND denominator
 *   admissible = (ra_lo::0 > 0 || ra_hi::0 < 0)
 *   If admissible: four-endpoint hull
 *     q1 = ra_lo::0 / ra_lo::0
 *     q2 = ra_lo::0 / ra_hi::0
 *     q3 = ra_hi::0 / ra_lo::0
 *     q4 = ra_hi::0 / ra_hi::0
 *   If inadmissible (zero in denominator interval): fallback to point denom
 *   lo_r = ite(admissible, min(q1..q4), lo_r_point)
 *   hi_r = ite(admissible, max(q1..q4), hi_r_point)
 *   ra_lo::1, ra_hi::1 pinned via RNE enclosure
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo::0   -- first div's lower bound declared
 *   ra_lo::1   -- second div's lifted lower bound declared
 *   (/ |smt_conv::ra_lo::0| |smt_conv::ra_lo::0|  -- q1: tracked/tracked
 *   (/ |smt_conv::ra_lo::0| |smt_conv::ra_hi::0|  -- q2: tracked/tracked
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
  double w = z / z; /* second RNE div: both operands tracked */

  /* Always false in real/integer encoding: w == z / z exactly. */
  assert(w != z / z);
  return 0;
}
