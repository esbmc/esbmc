/* Multi-if body where a later IF guard depends on a prior assignment
 * through a widening cast.
 *
 * Block 0: x = x - 1
 * Block 1: if ((long long)x >= 0) y = y - 1;
 *
 * The guard at block 1 is `(int64)x >= 0`, evaluated AFTER `x = x - 1`,
 * so the substituted path-condition becomes `(int64)(x - 1) >= 0`, i.e.
 * pre-state `x >= 1`. The cast peeling in measure_from_relational and
 * the apply_body path-cond substitution must compose correctly so the
 * path the THEN-arm runs is recognised as having pre-state x >= 1 (a
 * usable fact for the synthesizer) rather than the looser `x >= 0`.
 *
 * The loop guard is `y > 0`; rank m = y, body decreases y on the
 * then-arm. The else-arm doesn't decrease y but the rank check is
 * symmetric -- the rank derived from `y > 0` decreases on every path
 * regardless of x because y is only updated in the then-arm (else-arm
 * leaves y alone, m'=m, which would FAIL the strict-decrease check).
 * To make the test certifiable we ensure y always decreases:
 *
 *     while (y > 0) {
 *         x = x - 1;
 *         if ((long long)x >= 0) y = y - 1;
 *         else                   y = y - 1;
 *     }
 *
 * That trivially terminates (both arms decrement y by 1). Expected:
 * VERIFICATION SUCCESSFUL. The point of the test is structural -- it
 * exercises (i) typecast-peeling on the inner IF guard, (ii) apply_body
 * substitution of the prior `x = x - 1` into the IF guard for the
 * path-condition, (iii) per-path strict decrease on a 2-path body.
 */

extern int __VERIFIER_nondet_int(void);

int main()
{
  int x = __VERIFIER_nondet_int();
  int y = __VERIFIER_nondet_int();
  while (y > 0)
  {
    x = x - 1;
    if ((long long)x >= 0)
      y = y - 1;
    else
      y = y - 1;
  }
  return 0;
}
