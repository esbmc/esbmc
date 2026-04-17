/* Regression test: RNE interval lifting for ieee_mul -- both operands tracked,
 * single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-mul-both-tracked but exercises the single-precision
 * (float) path. Verifies that both-tracked lookup fires and the mul hull is
 * computed from four endpoint products.
 *
 * PROOF SHAPE (B_near, RNE, single precision)
 * -------------------------------------------
 * First mul:  z = x * y   (both fresh -> point-interval fallback; stored)
 *   ir_ra_interval_map[real_z] = {ra_lo::0, ra_hi::0}
 *
 * Second mul:  w = z * z  (both operands tracked -> full lift)
 *   p1 = ra_lo::0 * ra_lo::0
 *   p2 = ra_lo::0 * ra_hi::0  (= p3)
 *   p4 = ra_hi::0 * ra_hi::0
 *   lo_r = min(p1,p2,p3,p4),  hi_r = max(p1,p2,p3,p4)
 *   ra_lo::1, ra_hi::1 pinned with B_near (single eps)
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo::0   -- first mul's interval lower bound declared
 *   ra_lo::1   -- second mul's lifted lower bound declared
 *   (* |smt_conv::ra_lo::0|  -- endpoint product in hull computation
 *   (ite        -- nested ITE present
 *   5960464477539063  -- Z3 numerator for eps_rel = 2^-24 (single)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern float __VERIFIER_nondet_float(void);

int main(void)
{
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x * y; /* first RNE mul: both fresh -> point fallback; stored */
  float w = z * z; /* second RNE mul: both operands tracked -> full lift */

  /* Always false in real/integer encoding: w == z * z exactly. */
  assert(w != z * z);
  return 0;
}
