/*
 * Two function calls in the middle of the && chain (extends GitHub #3735).
 * Invariant: qm, qn ∈ {0,1} and qm ≠ qn; in_range(qm) and in_range(qn)
 * hold; inc() preserves the range and qm ≠ qn.  Expected: VERIFICATION SUCCESSFUL.
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
    0 <= qm && qm < 2 && /* expression guard     */
    in_range(qm) &&      /* FC #1 in MIDDLE      */
    in_range(qn) &&      /* FC #2 in MIDDLE      */
    (qm != qn)
  );
  for (int i = 0; i < 10; i++)
  {
    qm = inc(qm);
    qn = inc(qn);
  }

  return 0;
}
