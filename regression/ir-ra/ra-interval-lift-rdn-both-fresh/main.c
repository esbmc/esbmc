/* Regression test: RDN (ROUND_TO_MINUS_INF) interval lifting for ieee_add --
 * both operands fresh, double precision.
 *
 * PURPOSE
 * -------
 * Verifies that theorem-driven interval lifting fires for ieee_add under
 * ROUND_TO_MINUS_INF when both operands are fresh nondet variables.
 * With no tracked operands both use the point-interval fallback {t, t},
 * so the hull degenerates to the single-step RDN enclosure:
 *   LR = UR = x + y
 *   ra_lo_dn::0 = LR - B_dir(LR)
 *   ra_hi_dn::0 = UR        (exact upper: RDN never rounds above true value)
 *
 * PROOF SHAPE (B_dir, RDN, double precision)
 * ------------------------------------------
 * EbRDN([LR, UR]) = [LR - B_dir(LR), UR]
 * B_dir(r) = eps_rel_dir * |r| + eps_abs,  eps_rel_dir = 2^-52
 *
 * PATTERNS CHECKED (see test.desc)
 * ---------------------------------
 *   ra_lo_dn::[0-9]+  -- RDN interval lower bound declared
 *   ra_hi_dn::[0-9]+  -- RDN interval upper bound declared
 *   (ite               -- absolute value in B_dir computation
 *   22204460492503131  -- Z3 numerator for eps_rel_dir = 2^-52 (double)
 *   ^VERIFICATION FAILED$
 */
#include <assert.h>

extern int __ESBMC_rounding_mode;
extern double __VERIFIER_nondet_double(void);

int main(void)
{
  __ESBMC_rounding_mode = 3; /* ROUND_TO_MINUS_INF */
  double x = __VERIFIER_nondet_double();
  double y = __VERIFIER_nondet_double();
  double z = x + y; /* RDN add: both fresh -> point fallback */

  /* Always false in real/integer encoding. */
  assert(z != x + y);
  return 0;
}
