/* Regression test: RNA (ROUND_TO_AWAY) interval lifting for ieee_mul --
 * both operands tracked, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-rna-mul-both-tracked but exercises the
 * single-precision (float) path. Verifies that both-tracked lookup fires
 * and the mul hull is computed from four endpoint products under RNA.
 *
 * PROOF SHAPE (B_near, RNA, single precision)
 * -------------------------------------------
 * First mul:  z = x * y   (both fresh -> point-interval fallback; stored)
 *   ir_ra_interval_map[real_z] = {ra_lo_aw::0, ra_hi_aw::0}
 *
 * Second mul:  w = z * z  (both operands tracked -> full lift)
 *   p1 = ra_lo_aw::0 * ra_lo_aw::0
 *   p2 = ra_lo_aw::0 * ra_hi_aw::0  (= p3)
 *   p4 = ra_hi_aw::0 * ra_hi_aw::0
 *   lo_r = min(p1,p2,p3,p4),  hi_r = max(p1,p2,p3,p4)
 *   ra_lo_aw::1, ra_hi_aw::1 pinned with B_near (single eps, same as RNE)
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_aw::0   -- first mul's interval lower bound declared
 *   ra_lo_aw::1   -- second mul's lifted lower bound declared
 *   (* |smt_conv::ra_lo_aw::0|  -- endpoint product in hull computation
 *   (ite           -- nested ITE present
 *   5960464477539063  -- Z3 numerator for eps_rel = 2^-24 (single, same as RNE)
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
  float z = x * y; /* first RNA mul: both fresh -> point fallback; stored */
  float w = z * z; /* second RNA mul: both operands tracked -> full lift */

  /* Always false in real/integer encoding: w == z * z exactly. */
  assert(w != z * z);
  return 0;
}
