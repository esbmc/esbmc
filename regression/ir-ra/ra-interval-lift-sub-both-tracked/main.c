/* Regression test: RNE interval lifting for ieee_sub -- both operands tracked.
 *
 * PURPOSE
 * -------
 * Verifies that when both operands of a second RNE ieee_sub were themselves
 * results of a prior tracked RNE ieee_sub, ir_ra_interval_map lookup fires
 * for both operands and the interval-lifted subtraction path is taken.
 *
 * PROOF SHAPE (B_near, RNE, double precision)
 * -------------------------------------------
 * First sub:  z = x - y   (both fresh -> point-interval fallback)
 *   iv(x) = {x, x},  iv(y) = {y, y}
 *   L_R1 = x - y = real_z,   U_R1 = x - y = real_z
 *   ra_lo::0 = real_z - B_near(real_z)
 *   ra_hi::0 = real_z + B_near(real_z)
 *   stored: ir_ra_interval_map[real_z] = {ra_lo::0, ra_hi::0}
 *
 * Second sub:  w = z - z  (both operands are z -> both tracked)
 *   iv(z) = {ra_lo::0, ra_hi::0}  (found in map, same entry twice)
 *   L_R2 = ra_lo::0 - ra_hi::0
 *   U_R2 = ra_hi::0 - ra_lo::0
 *   ra_lo::1 = L_R2 - B_near(L_R2)
 *   ra_hi::1 = U_R2 + B_near(U_R2)
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo::0   -- first subtraction's interval lower bound declared
 *   ra_lo::1   -- second subtraction's lifted lower bound declared
 *   (ite        -- absolute value present in B_near computations
 *   5551115123125783  -- Z3 numerator for eps_rel = 2^-53 (double)
 *   ^VERIFICATION FAILED$  -- run completed
 */
#include <assert.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x - y;   /* first RNE sub: both fresh -> point fallback; stored */
  double w = z - z;   /* second RNE sub: both operands tracked -> full lift */

  /* Always false in real/integer encoding: w == z - z exactly. */
  assert(w != z - z);
  return 0;
}
