/* A side-effecting loop guard. The frontend hoists `cnt--` above the point the
 * schema havocs at, so the exit edge tests a pre-havoc temporary, is
 * infeasible, and the assertion after the loop is never reached -- an
 * unreached claim is silently not a failing one. Reduced from
 * regression/k-induction/github_1092_6_false, where this reported VERIFICATION
 * SUCCESSFUL. Houdini must notice it lost a claim and decline to call it a
 * proof. Reproducible without Houdini with a hand-written invariant under
 * --loop-invariant-check, so the schema defect is the underlying one. */
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
