/* Regression test: RNA (ROUND_TO_AWAY) interval lifting for ieee_div --
 * both operands tracked, double precision.
 *
 * PURPOSE
 * -------
 * Verifies proof-aligned compositional interval lifting for RNA ieee_div
 * when both the numerator and denominator are prior tracked results.
 * The four-endpoint hull formula divides tracked-over-tracked endpoint
 * pairs in the admissible branch (denominator interval excludes zero).
 *
 * PROOF SHAPE (B_near, RNA, double precision)
 * -------------------------------------------
 * First div:  z = x / y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[z_smt] = {ra_lo_aw::0, ra_hi_aw::0}
 *
 * Second div:  w = z / z  (both operands tracked)
 *   iv(z) = {ra_lo_aw::0, ra_hi_aw::0} for numerator AND denominator
 *   admissible = (ra_lo_aw::0 > 0 || ra_hi_aw::0 < 0)
 *   If admissible: four-endpoint hull
 *     q1 = ra_lo_aw::0 / ra_lo_aw::0
 *     q2 = ra_lo_aw::0 / ra_hi_aw::0
 *     q3 = ra_hi_aw::0 / ra_lo_aw::0
 *     q4 = ra_hi_aw::0 / ra_hi_aw::0
 *   lo_r = ite(admissible, min(q1..q4), lo_r_point)
 *   hi_r = ite(admissible, max(q1..q4), hi_r_point)
 *   ra_lo_aw::1, ra_hi_aw::1 pinned via RNA enclosure
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_aw::0   -- first div's lower bound declared
 *   ra_lo_aw::1   -- second div's lifted lower bound declared
 *   (/ |smt_conv::ra_lo_aw::0| |smt_conv::ra_lo_aw::0|  -- q1
 *   (/ |smt_conv::ra_lo_aw::0| |smt_conv::ra_hi_aw::0|  -- q2
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
  double w = z / z; /* second RNA div: both operands tracked */

  /* Always false in real/integer encoding: w == z / z exactly. */
  assert(w != z / z);
  return 0;
}
