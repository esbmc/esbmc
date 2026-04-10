/* Regression test: RUP (ROUND_TO_PLUS_INF) interval lifting for ieee_add --
 * both operands tracked, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-rup-both-tracked but exercises the single-precision
 * (float) path. Verifies that the get_single_eps_up() /
 * get_single_min_subnormal() branch in apply_ieee754_rup_enclosure is reached
 * when both operands of a second RUP ieee_add are tracked.
 *
 * PROOF SHAPE (B_dir, RUP, single precision)
 * ------------------------------------------
 * Second add:  w = z + z  (both operands tracked -> full lift)
 *   L_R2 = ra_lo_up::0 + ra_lo_up::0
 *   U_R2 = ra_hi_up::0 + ra_hi_up::0
 *   ra_lo_up::1 = L_R2          (exact lower: RUP never rounds below)
 *   ra_hi_up::1 = U_R2 + B_dir(U_R2)   [single eps: 2^-23]
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_up::0   -- first addition's interval lower bound declared
 *   ra_lo_up::1   -- second addition's lifted lower bound declared
 *   (+ ra_lo_up::0 ra_lo_up::0)  -- L_R2 subterm confirming tracked lookup
 *   (ite           -- absolute value present in B_dir computation
 *   8388608        -- Z3 denominator for eps_rel_dir = 2^-23 (single)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern float __VERIFIER_nondet_float(void);

int main(void)
{
  __ESBMC_rounding_mode = 2; /* ROUND_TO_PLUS_INF */
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x + y;   /* first RUP add: both fresh -> point fallback; stored */
  float w = z + z;   /* second RUP add: both operands tracked -> full lift */

  /* Always false in real/integer encoding: w == z + z exactly. */
  assert(w != z + z);
  return 0;
}
