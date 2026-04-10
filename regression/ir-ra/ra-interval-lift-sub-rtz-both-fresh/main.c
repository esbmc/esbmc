/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_sub --
 * both operands fresh, double precision.
 *
 * PURPOSE
 * -------
 * Verifies that theorem-driven interval lifting fires for ieee_sub under
 * ROUND_TO_ZERO when both operands are fresh nondet variables.
 * With no tracked operands both use the point-interval fallback {t, t},
 * so the hull degenerates to lo_r = hi_r = real_result (single point).
 * The RTZ enclosure collapses to the single-step sign-dependent shape:
 *
 *   r >= 0: ra_lo_tz = r - B_dir(r),  ra_hi_tz = r  (truncate down)
 *   r <  0: ra_lo_tz = r,             ra_hi_tz = r + B_dir(r)  (truncate up)
 *
 * encoded as nested ITE on the sign of r.
 *
 * PROOF SHAPE (B_dir, RTZ, double precision)
 * ------------------------------------------
 * B_dir(r) = eps_rel_dir * |r| + eps_abs,  eps_rel_dir = 2^-52
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_tz::[0-9]+  -- RTZ interval lower bound declared
 *   ra_hi_tz::[0-9]+  -- RTZ interval upper bound declared
 *   (ite               -- sign-conditional ITE in enclosure formula
 *   22204460492503131  -- Z3 numerator for eps_rel_dir = 2^-52 (double)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 4; /* ROUND_TO_ZERO */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x - y; /* RTZ sub: both fresh -> point fallback */

  /* Always false in real/integer encoding. */
  assert(z != x - y);
  return 0;
}
