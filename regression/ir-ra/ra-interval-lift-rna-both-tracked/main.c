/* Regression test: RNA (ROUND_TO_AWAY) interval lifting -- both operands tracked.
 *
 * PURPOSE
 * -------
 * Verifies that when both operands of a second RNA ieee_add were themselves
 * results of a prior tracked RNA ieee_add, ir_ra_interval_map lookup fires
 * for both operands and the interval-lifted RNA path is taken.
 *
 * PROOF SHAPE (B_near, RNA, double precision)
 * -------------------------------------------
 * First add:  z = x + y   (both fresh -> point-interval fallback)
 *   iv(x) = {x, x},  iv(y) = {y, y}
 *   L_R1 = x + y = real_z,   U_R1 = x + y = real_z
 *   ra_lo_aw::0 = real_z - B_near(real_z)
 *   ra_hi_aw::0 = real_z + B_near(real_z)
 *   stored: ir_ra_interval_map[real_z] = {ra_lo_aw::0, ra_hi_aw::0}
 *
 * Second add:  w = z + z  (both operands are z -> both tracked)
 *   iv(z) = {ra_lo_aw::0, ra_hi_aw::0}  (found in map, same entry twice)
 *   L_R2 = ra_lo_aw::0 + ra_lo_aw::0
 *   U_R2 = ra_hi_aw::0 + ra_hi_aw::0
 *   ra_lo_aw::1 = L_R2 - B_near(L_R2)
 *   ra_hi_aw::1 = U_R2 + B_near(U_R2)
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_aw::0   -- first addition's interval lower bound declared
 *   ra_lo_aw::1   -- second addition's lifted lower bound declared
 *   (ite           -- absolute value present in B_near computations
 *   5551115123125783  -- Z3 numerator for eps_rel = 2^-53 (double, same as RNE)
 *   ^VERIFICATION FAILED$  -- run completed
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 1; /* ROUND_TO_AWAY */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x + y;   /* first RNA add: both fresh -> point fallback; stored */
  double w = z + z;   /* second RNA add: both operands tracked -> full lift */

  /* Always false in real/integer encoding: w == z + z exactly. */
  assert(w != z + z);
  return 0;
}
