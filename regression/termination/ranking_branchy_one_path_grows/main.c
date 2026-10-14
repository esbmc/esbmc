/* Soundness guard for the branchy-body extension: both paths of an
 * if/else in the loop body must strictly decrease the measure, not just
 * one. Here the then-arm decrements i (m = 255 - i decreases by 1) but
 * the else-arm increments it (m increases by 1), so the loop does not
 * provably terminate (it can stay above the guard forever by picking the
 * else-arm). recognize_loop accepts the shape and yields two paths; the
 * decrease obligation on the else-path is satisfiable (m'_else = m + 1
 * >= m holds), so the prover must NOT certify the loop and control must
 * fall through to the existing machinery, leaving the verdict UNKNOWN.
 *
 * Expected verdict: VERIFICATION UNKNOWN. */

extern int __VERIFIER_nondet_int(void);

int main()
{
  int i = __VERIFIER_nondet_int();
  while (i < 255)
  {
    if (__VERIFIER_nondet_int() != 0)
      i = i + 1; /* progress toward guard becoming false */
    else
      i = i - 1; /* moves AWAY from the guard becoming false */
  }
  return 0;
}
