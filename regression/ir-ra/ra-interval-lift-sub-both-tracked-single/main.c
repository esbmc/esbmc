/* Regression test: RNE interval lifting for ieee_sub -- both operands tracked,
 * single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-sub-both-tracked but exercises the single-precision
 * (float) path. Verifies that the get_single_eps_rel() /
 * get_single_min_subnormal() branch in apply_ieee754_rne_interval_add is
 * reached when both operands of a second RNE ieee_sub are tracked.
 *
 * PROOF SHAPE (B_near, RNE, single precision)
 * -------------------------------------------
 * First sub:  z = x - y   (both fresh -> point-interval fallback)
 *   ra_lo::0 = real_z - B_near(real_z)
 *   stored: ir_ra_interval_map[real_z] = {ra_lo::0, ra_hi::0}
 *
 * Second sub:  w = z - z  (both operands tracked -> full lift)
 *   L_R2 = ra_lo::0 - ra_hi::0
 *   U_R2 = ra_hi::0 - ra_lo::0
 *   ra_lo::1 = L_R2 - B_near(L_R2)   [single eps: 2^-24]
 *   ra_hi::1 = U_R2 + B_near(U_R2)
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo::0   -- first subtraction's interval lower bound declared
 *   ra_lo::1   -- second subtraction's lifted lower bound declared
 *   (ite        -- absolute value present in B_near computations
 *   5960464477539063  -- Z3 numerator for eps_rel = 2^-24 (single)
 *   ^VERIFICATION FAILED$  -- run completed
 */
#include <assert.h>

extern float __VERIFIER_nondet_float(void);

int main(void)
{
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x - y;   /* first RNE sub: both fresh -> point fallback; stored */
  float w = z - z;   /* second RNE sub: both operands tracked -> full lift */

  /* Always false in real/integer encoding: w == z - z exactly. */
  assert(w != z - z);
  return 0;
}
