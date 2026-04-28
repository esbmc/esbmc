/*
 * Regression test for GitHub issue #3735 -- FAIL variant.
 *
 * Same structure as the pass case (function call in the MIDDLE of the &&
 * chain, triggering Pattern B short-circuit encoding), but the invariant
 * makes a wrong claim ("qm == 0") that holds at loop entry yet is NOT
 * preserved by the loop body.
 *
 * Expected: VERIFICATION FAILED at the inductive step.
 *   - Base case: qm=0, qn=1.  "qm==0 && in_range(1) && 0!=1" holds. PASS.
 *   - Inductive step: after inc(qm), qm becomes 1. "qm==0" is false. FAIL.
 *
 * This verifies that the Pattern B fix does not make ESBMC too permissive:
 * genuinely wrong invariants must still be detected.
 */

#define MAX 2

int inc(int qa)
{
  int res = (qa + 1) % MAX;
  return res;
}

int in_range(const int qa)
{
  return 0 <= qa && qa < MAX;
}

int main()
{
  int qm = 0;
  int qn = 1; /* (0 + 1) % MAX */

  /* Wrong invariant: "qm == 0" is not preserved by inc(qm).
   * in_range(qn) is intentionally placed in the MIDDLE of the && chain
   * to exercise Pattern B (short-circuit) handling. */
  __ESBMC_loop_invariant(
    qm == 0 &&        /* fails after first inc: qm becomes 1  */
    in_range(qn) &&   /* function in MIDDLE                   */
    (qm != qn)
  );
  for (int count = 0; count < 10; count++)
  {
    qm = inc(qm);
    qn = inc(qn);
  }

  return 0;
}
