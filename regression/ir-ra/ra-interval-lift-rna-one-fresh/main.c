/* Regression test: RNA (ROUND_TO_AWAY) interval lifting -- one tracked, one fresh.
 *
 * PURPOSE
 * -------
 * Verifies the mixed lookup path for RNA: when one operand of a second
 * RNA ieee_add is tracked in ir_ra_interval_map and the other is a fresh
 * nondet variable, the tracked operand uses its stored interval while the
 * fresh one falls back to the point interval {side, side}.
 *
 * PROOF SHAPE (B_near, RNA, double precision)
 * -------------------------------------------
 * First add:  z = x + y   (both fresh -> point fallback; z stored in map)
 *   ir_ra_interval_map[real_z] = {ra_lo_aw::0, ra_hi_aw::0}
 *
 * Second add:  w = z + x  (z tracked, x fresh -> mixed path)
 *   iv(z) = {ra_lo_aw::0, ra_hi_aw::0}   (from map)
 *   iv(x) = {x_smt, x_smt}               (point fallback)
 *   L_R2 = ra_lo_aw::0 + x_smt
 *   U_R2 = ra_hi_aw::0 + x_smt
 *   ra_lo_aw::1 = L_R2 - B_near(L_R2)
 *   ra_hi_aw::1 = U_R2 + B_near(U_R2)
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_aw::0   -- first addition's interval lower bound declared
 *   ra_lo_aw::1   -- second addition's mixed-path lower bound declared
 *   (ite           -- absolute value present
 *   5551115123125783  -- Z3 numerator for eps_rel = 2^-53 (double)
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
  double w = z + x;   /* second RNA add: z tracked, x fresh -> mixed path */

  /* Always false in real/integer encoding: w == z + x exactly. */
  assert(w != z + x);
  return 0;
}
