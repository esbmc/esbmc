/*
 * Pin contract: the simplifier folds `0 << x` (and `0 >> x`, `0 >>s x`) to 0,
 * but the --ub-shift-check assertions on the shift amount must still fail
 * for nondeterministic n that may be negative or >= width.
 *
 * Two levels of defense make this work:
 *  1. UB-check assertions are inserted during goto-program processing,
 *     before symex/simplifier run.
 *  2. expr2t::simplify short-circuits on overflow2t (no descent), so the
 *     simplifier cannot fold the shift inside the overflow check.
 */
int nondet_int();

int main() {
  int n = nondet_int();
  // n could be negative or >= 32 — the shift is UB even though lhs is 0.
  int v = 0 << n;
  (void)v;
  return 0;
}
