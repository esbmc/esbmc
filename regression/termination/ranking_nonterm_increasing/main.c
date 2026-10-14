/* Soundness guard for the ranking-function checker.
 *
 * `while (n > 0) n = n + 1` does NOT terminate: under the guard n > 0,
 * the counter increases without bound (until signed overflow, which is
 * UB). The tempting measure m = n (the guard difference n - 0) is the
 * same one the checker would build, but it INCREASES, so the decrease
 * obligation `guard AND (m' >= m)` is satisfiable (not UNSAT) and the
 * checker must refuse to certify termination.
 *
 * The ranking checker must therefore NOT print "Ranking function shows
 * all executions terminate" here; control falls through to the existing
 * marker / forward-condition / inductive-step machinery, which leaves
 * this case as VERIFICATION UNKNOWN. A wrong certification would print
 * SUCCESSFUL — this test fails loudly if the checker ever does that.
 *
 * Expected verdict: VERIFICATION UNKNOWN. */

extern int __VERIFIER_nondet_int(void);

int main()
{
  int n = __VERIFIER_nondet_int();
  while (n > 0)
    n = n + 1;
  return 0;
}
