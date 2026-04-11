/* Regression test: RNE interval lifting for ieee_mul -- both operands tracked.
 *
 * PURPOSE
 * -------
 * Verifies that when both operands of a second RNE ieee_mul were themselves
 * results of a prior tracked RNE ieee_mul, ir_ra_interval_map lookup fires
 * for both operands and the interval-lifted multiplication path is taken.
 *
 * PROOF SHAPE (B_near, RNE, double precision)
 * -------------------------------------------
 * First mul:  z = x * y   (both fresh -> point-interval fallback)
 *   iv(x) = {x, x},  iv(y) = {y, y}
 *   p1=p2=p3=p4 = x * y = real_z
 *   ra_lo::0 = real_z - B_near(real_z)
 *   ra_hi::0 = real_z + B_near(real_z)
 *   stored: ir_ra_interval_map[real_z] = {ra_lo::0, ra_hi::0}
 *
 * Second mul:  w = z * z  (both operands are z -> both tracked)
 *   iv(z) = {ra_lo::0, ra_hi::0}  (found in map, same entry twice)
 *   p1 = ra_lo::0 * ra_lo::0
 *   p2 = ra_lo::0 * ra_hi::0
 *   p3 = ra_hi::0 * ra_lo::0
 *   p4 = ra_hi::0 * ra_hi::0
 *   lo_r = min(p1,p2,p3,p4) via nested ITE
 *   hi_r = max(p1,p2,p3,p4) via nested ITE
 *   ra_lo::1 = lo_r - B_near(lo_r)
 *   ra_hi::1 = hi_r + B_near(hi_r)
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo::0   -- first multiplication's interval lower bound declared
 *   ra_lo::1   -- second multiplication's lifted lower bound declared
 *   (* |smt_conv::ra_lo::0|  -- endpoint product in hull computation
 *   (ite        -- nested ITE for min/max hull and absolute value
 *   5551115123125783  -- Z3 numerator for eps_rel = 2^-53 (double)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern double __VERIFIER_nondet_double(void);

int main(void)
{
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x * y; /* first RNE mul: both fresh -> point fallback; stored */
  double w = z * z; /* second RNE mul: both operands tracked -> full lift */

  /* Always false in real/integer encoding: w == z * z exactly. */
  assert(w != z * z);
  return 0;
}
