/* Regression test: RUP (ROUND_TO_PLUS_INF) interval lifting for ieee_add --
 * both operands fresh, double precision.
 *
 * PURPOSE
 * -------
 * Verifies that theorem-driven interval lifting fires for ieee_add under
 * ROUND_TO_PLUS_INF when both operands are fresh nondet variables.
 * With no tracked operands both use the point-interval fallback {t, t},
 * so the hull degenerates to the single-step RUP enclosure:
 *   LR = UR = x + y
 *   ra_lo_up::0 = LR        (exact lower: RUP never rounds below true value)
 *   ra_hi_up::0 = UR + B_dir(UR)
 *
 * PROOF SHAPE (B_dir, RUP, double precision)
 * ------------------------------------------
 * EbRUP([LR, UR]) = [LR, UR + B_dir(UR)]
 * B_dir(r) = eps_rel_dir * |r| + eps_abs,  eps_rel_dir = 2^-52
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_up::[0-9]+  -- RUP interval lower bound declared
 *   ra_hi_up::[0-9]+  -- RUP interval upper bound declared
 *   (ite               -- absolute value in B_dir computation
 *   22204460492503131  -- Z3 numerator for eps_rel_dir = 2^-52 (double)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 2; /* ROUND_TO_PLUS_INF */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x + y; /* RUP add: both fresh -> point fallback */

  /* Always false in real/integer encoding. */
  assert(z != x + y);
  return 0;
}
