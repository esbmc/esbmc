/* A side-effecting loop guard, reduced from
 * regression/k-induction/github_1092_6_false, where this reported VERIFICATION
 * SUCCESSFUL. The schema used to hoist `cnt--` above the havoc point, so the
 * exit edge tested a pre-havoc temporary, was infeasible, and the assertion
 * after the loop was never reached -- an unreached claim is silently not a
 * failing one. This test pinned Houdini's unchecked-claim guard noticing that
 * and declining to call it a proof.
 *
 * PR #7482 fixed the underlying schema defect (issue #7478), so the claim
 * survives and the real bug is reported directly. The guard is consequently no
 * longer exercised here; whether it still has a reachable case of its own is
 * worth deciding separately rather than assuming from this file. */
#include <assert.h>

int main()
{
  int cnt = 4;

  while (cnt--)
  {
  }

  assert(0);
  return 0;
}
