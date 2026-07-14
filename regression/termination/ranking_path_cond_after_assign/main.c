/* Soundness pin for path-condition substitution against prior-block
 * assignments.
 *
 * The body has two sequential blocks: an assignment to `y`, then an `if
 * (y > 0)`. The IF guard `y > 0` is evaluated AFTER `y = y - 1` runs,
 * so the path-condition for the IF's then-arm reflects the post-`y-1`
 * value of y, not the pre-iteration value. recognize_loop captures this
 * by calling apply_body on the prior-block assignments to the IF guard
 * before recording the path-cond atom.
 *
 * Without that substitution the prover would record `y > 0` as a
 * pre-state constraint on the path, which is a strictly stronger
 * predicate than the actual `(y - 1) > 0` (i.e. pre-state `y > 1`) —
 * leading the prover to deem some feasible paths infeasible and
 * vacuously discharge their decrease obligations. Here the path where
 * the then-arm runs (decrementing x) is feasible when y >= 2 pre-state,
 * but a buggy path-cond `y > 0` and a non-decreasing alternative arm
 * could cause unsound certification on a non-terminating loop.
 *
 * The loop below is non-terminating for some inputs (start y = 1: after
 * `y = y - 1`, y becomes 0; the then-arm with `y > 0` after assign is
 * false, so the empty else-arm runs, x unchanged, and the back-edge
 * goes around with y now stuck at -1 — but the loop guard x > 0 keeps
 * holding because x never changed). The prover must NOT certify.
 *
 * Expected verdict: VERIFICATION UNKNOWN. */

extern int __VERIFIER_nondet_int(void);

int main()
{
  int x = __VERIFIER_nondet_int();
  int y = __VERIFIER_nondet_int();
  while (x > 0)
  {
    y = y - 1;
    if (y > 0)
      x = x - 1;
    /* else: nothing; x unchanged → loop guard x > 0 still holds. */
  }
  return 0;
}
