/* Compound `&&` loop guard for the ranking checker.
 *
 * `while (x < N && y > 0)` exits as soon as EITHER conjunct becomes
 * false, so making EITHER rank-derived measure reach its floor proves
 * termination. measure_candidates_from_guard flattens the `&&` and
 * yields one (m, L) per relational conjunct; prove_loop_terminates tries
 * each and succeeds when one discharges its bounded-below and strict-
 * decrease obligations under the full guard. Here the second conjunct
 * `y > 0` with measure y and body `y = y - 1` does the job.
 *
 * Expected verdict: VERIFICATION SUCCESSFUL. */

extern int __VERIFIER_nondet_int(void);

int main()
{
  int x = __VERIFIER_nondet_int();
  int y = __VERIFIER_nondet_int();
  while (x < 1000 && y > 0)
  {
    /* x might not change here, but y always decreases — and the loop
     * exits as soon as y reaches zero. */
    y = y - 1;
  }
  return 0;
}
