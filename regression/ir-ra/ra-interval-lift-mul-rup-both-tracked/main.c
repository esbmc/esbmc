/* Regression test: RUP (ROUND_TO_PLUS_INF) interval lifting for ieee_mul --
 * both operands tracked, double precision.
 *
 * PURPOSE
 * -------
 * Verifies that when both operands of a second RUP ieee_mul were themselves
 * results of a prior tracked RUP ieee_mul, ir_ra_interval_map lookup fires
 * for both operands and the interval-lifted RUP multiplication path is taken.
 *
 * PROOF SHAPE (B_dir, RUP, double precision)
 * ------------------------------------------
 * First mul:  z = x * y   (both fresh -> point-interval fallback)
 *   iv(x) = {x, x},  iv(y) = {y, y}
 *   p1=p2=p3=p4 = x * y = real_z
 *   ra_lo_up::0 = real_z          (exact lower)
 *   ra_hi_up::0 = real_z + B_dir(real_z)
 *   stored: ir_ra_interval_map[real_z] = {ra_lo_up::0, ra_hi_up::0}
 *
 * Second mul:  w = z * z  (both operands are z -> both tracked)
 *   iv(z) = {ra_lo_up::0, ra_hi_up::0}  (found in map, same entry twice)
 *   p1 = ra_lo_up::0 * ra_lo_up::0
 *   p2 = ra_lo_up::0 * ra_hi_up::0
 *   p3 = ra_hi_up::0 * ra_lo_up::0
 *   p4 = ra_hi_up::0 * ra_hi_up::0
 *   lo_r = min(p1,p2,p3,p4) via nested ITE
 *   hi_r = max(p1,p2,p3,p4) via nested ITE
 *   ra_lo_up::1 = lo_r          (exact lower)
 *   ra_hi_up::1 = hi_r + B_dir(hi_r)
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_up::0   -- first mul's interval lower bound declared
 *   ra_lo_up::1   -- second mul's lifted lower bound declared
 *   (* |smt_conv::ra_lo_up::0|  -- endpoint product in hull computation
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
  double w = z * z; /* second RUP mul: both operands tracked -> full lift */

  /* Always false in real/integer encoding: w == z * z exactly. */
  assert(w != z * z);
  return 0;
}
