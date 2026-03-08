/*
 * KNOWNBUG: Two function calls in the middle of an && chain.
 *
 * Pattern B (short-circuit) handling currently supports only ONE function
 * call in the middle of the && chain (GitHub #3735).  When a second FC
 * appears, the backward walk breaks at the second FC, the guard condition
 * for the first FC is left as a stale pre-havoc value, and the outer
 * expression guard (0 <= qm && qm < 2) is never recovered.
 *
 * The invariant below IS correct:
 *   qm, qn ∈ {0,1} and qm ≠ qn at all times.
 *   in_range(qm) and in_range(qn) are both always true.
 *   inc() cycles values within {0,1} so qm ≠ qn is preserved.
 *
 * Expected: VERIFICATION SUCCESSFUL
 * Actual:   VERIFICATION FAILED (inductive step, due to incomplete guard
 *           reconstruction for the second FC).
 */

#define MAX 2

int in_range(const int qa)
{
  return 0 <= qa && qa < MAX;
}

int inc(int qa)
{
  int res = (qa + 1) % MAX;
  return res;
}

int main()
{
  int qm = nondet_bool(); /* qm ∈ {0, 1} */
  int qn = 1 - qm;       /* qn = 1-qm, so qm ≠ qn */

  __ESBMC_loop_invariant(
    0 <= qm && qm < 2 && /* expression guard            */
    in_range(qm) &&      /* FC #1 in MIDDLE             */
    in_range(qn) &&      /* FC #2 in MIDDLE (not fixed) */
    (qm != qn)
  );
  for (int i = 0; i < 10; i++)
  {
    qm = inc(qm);
    qn = inc(qn);
  }

  return 0;
}
