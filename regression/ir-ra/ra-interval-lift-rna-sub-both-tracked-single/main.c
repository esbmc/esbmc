/* Regression test: RNA (ROUND_TO_AWAY) interval lifting for ieee_sub --
 * both operands tracked, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-rna-sub-both-tracked but exercises the
 * single-precision (float) path. Verifies that the get_single_eps_rel() /
 * get_single_min_subnormal() branch in apply_ieee754_rna_enclosure is
 * reached when both operands of a second RNA ieee_sub are tracked.
 *
 * PROOF SHAPE (B_near, RNA, single precision)
 * -------------------------------------------
 * First sub:  z = x - y   (both fresh -> point-interval fallback)
 *   ra_lo_aw::0 = real_z - B_near(real_z)
 *   stored: ir_ra_interval_map[real_z] = {ra_lo_aw::0, ra_hi_aw::0}
 *
 * Second sub:  w = z - z  (both operands tracked -> full lift)
 *   L_R2 = ra_lo_aw::0 - ra_hi_aw::0
 *   U_R2 = ra_hi_aw::0 - ra_lo_aw::0
 *   ra_lo_aw::1 = L_R2 - B_near(L_R2)   [single eps: 2^-24]
 *   ra_hi_aw::1 = U_R2 + B_near(U_R2)
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_aw::0   -- first subtraction's interval lower bound declared
 *   ra_lo_aw::1   -- second subtraction's lifted lower bound declared
 *   (- ra_lo_aw::0 ra_hi_aw::0)  -- L_R2 subterm confirming tracked lookup
 *   (ite           -- absolute value present in B_near computations
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
  float w = z - z;   /* second RNA sub: both operands tracked -> full lift */

  /* Always false in real/integer encoding: w == z - z exactly. */
  assert(w != z - z);
  return 0;
}
