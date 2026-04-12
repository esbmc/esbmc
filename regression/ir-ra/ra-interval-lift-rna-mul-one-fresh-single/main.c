/* Regression test: RNA (ROUND_TO_AWAY) interval lifting for ieee_mul --
 * one tracked, one fresh, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-rna-mul-one-fresh but exercises the
 * single-precision (float) path. Verifies the mixed lookup path: when one
 * operand of a second RNA ieee_mul is tracked and the other is fresh.
 *
 * PROOF SHAPE (B_near, RNA, single precision)
 * -------------------------------------------
 * Second mul:  w = z * x  (z tracked, x fresh -> mixed path)
 *   lo_r = min(ra_lo_aw::0 * x_smt, ra_hi_aw::0 * x_smt)
 *   hi_r = max(ra_lo_aw::0 * x_smt, ra_hi_aw::0 * x_smt)
 *   ra_lo_aw::1, ra_hi_aw::1 pinned with B_near (single eps, same as RNE)
 *
 * PATTERNS CHECKED (see test.desc)
 *   ra_lo_aw::0   -- first mul's interval lower bound declared
 *   ra_lo_aw::1   -- second mul's mixed-path lower bound declared
 *   (* |smt_conv::ra_lo_aw::0|  -- tracked endpoint in hull product
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
  float w = z * x; /* second RNA mul: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z * x exactly. */
  assert(w != z * x);
  return 0;
}
