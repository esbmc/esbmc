/* Regression: GitHub #7478 -- a do-while back edge carries the loop's own exit
 * test.  Cutting it with ASSUME(false) also killed the fall-through, deleting
 * every claim after the loop, and the inductive step was required of the
 * iteration that exits.  Compare k_induction_issue_dowhile_entry_cond, which
 * pins the same semantics for the combined pass (PR #3777). */
#include <assert.h>

int main()
{
  int x = 0;
  __ESBMC_loop_invariant(x <= 4);
  do
  {
    x++;
  } while (x < 5);
  assert(x == 5);
  return 0;
}
