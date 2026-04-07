/* Regression test: RNA (ROUND_TO_AWAY) interval lifting for ieee_sub --
 * one tracked, one fresh, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-rna-sub-one-fresh but exercises the
 * single-precision (float) path. Verifies the mixed lookup path: when one
 * operand of a second RNA ieee_sub is tracked and the other is a fresh
 * nondet variable, the tracked operand uses its stored interval while the
 * fresh one falls back to the point interval {side, side}.
 *
 * PROOF SHAPE (B_near, RNA, single precision)
 * -------------------------------------------
 * First sub:  z = x - y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[real_z] = {ra_lo_aw::0, ra_hi_aw::0}
 *
 * Second sub:  w = z - x  (z tracked, x fresh -> mixed path)
 *   iv(z) = {ra_lo_aw::0, ra_hi_aw::0}   (from map)
 *   iv(x) = {x_smt, x_smt}               (point fallback)
 *   L_R2 = ra_lo_aw::0 - x_smt
 *   U_R2 = ra_hi_aw::0 - x_smt
 *   ra_lo_aw::1 = L_R2 - B_near(L_R2)   [single eps: 2^-24]
 *   ra_hi_aw::1 = U_R2 + B_near(U_R2)
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_aw::0   -- first subtraction's interval lower bound declared
 *   ra_lo_aw::1   -- second subtraction's mixed-path lower bound declared
 *   (- ra_lo_aw::0 ...  -- L_R2 prefix confirming tracked lookup
 *   (ite           -- absolute value present
 *   5960464477539063  -- Z3 numerator for eps_rel = 2^-24 (single)
 *   ^VERIFICATION FAILED$  -- run completed
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern float __VERIFIER_nondet_float(void);

int main(void)
{
  __ESBMC_rounding_mode = 1; /* ROUND_TO_AWAY */
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x - y;   /* first RNA sub: both fresh -> point fallback; stored */
  float w = z - x;   /* second RNA sub: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z - x exactly. */
  assert(w != z - x);
  return 0;
}
