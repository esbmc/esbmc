/* Regression test: RTZ (ROUND_TO_ZERO) interval lifting for ieee_add --
 * both operands fresh, single precision.
 *
 * PURPOSE
 * -------
 * Mirrors ra-interval-lift-rtz-both-fresh but exercises the single-precision
 * (float) path. Verifies that when both operands are fresh nondet variables,
 * the point-interval fallback collapses to the single-step RTZ sign-conditional
 * formula with single-precision epsilon constants.
 *
 * PROOF SHAPE (B_dir, RTZ, single precision)
 * ------------------------------------------
 * Both x and y are fresh; lo_r == hi_r == real_z, producing:
 *   r >= 0: ra_lo_tz = r - B_dir(r),  ra_hi_tz = r   [eps_rel_dir = 2^-23]
 *   r <  0: ra_lo_tz = r,             ra_hi_tz = r + B_dir(r)
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_tz::       -- RTZ tight path taken (single precision)
 *   ra_hi_tz::       -- RTZ tight path taken
 *   (ite             -- sign-conditional ITE in enclosure formula
 *   8388608          -- Z3 denominator for eps_rel_dir = 2^-23 (single)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern float __VERIFIER_nondet_float(void);

int main(void)
{
  __ESBMC_rounding_mode = 4; /* ROUND_TO_ZERO */
  float x = __VERIFIER_nondet_float();
  float y = __VERIFIER_nondet_float();
  float z = x + y; /* RTZ add: both fresh -> point fallback */

  /* Always false in real/integer encoding. */
  assert(z != x + y);
  return 0;
}
