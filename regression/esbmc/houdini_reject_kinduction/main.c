/* --houdini-loop-invariants drives its own filtering rounds and applies the
 * loop-invariant schema itself, so it cannot share a run with another driver.
 * Layering it on k-induction is not merely redundant: goto_k_induction has
 * already havoc'd the loops by the time the strategy runs, and on
 * regression/k-induction/trex02_bug that combination reported VERIFICATION
 * SUCCESSFUL for a program both modes report FAILED on individually. */
#include <assert.h>

int main()
{
  float x = 2;

  while (nondet_bool())
    x = 2 * x - 1;

  assert(x > 0);
  return 0;
}
