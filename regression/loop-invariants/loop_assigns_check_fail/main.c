/*
 * loop_assigns_check_fail: Loop frame rule assigns compliance — violation case.
 *
 * The loop declares __ESBMC_loop_assigns(i), claiming only i is modified.
 * However, the loop body ALSO modifies j — violating the assigns clause.
 *
 * The loop invariant intentionally only mentions i (not j), so the inductive
 * invariant check alone cannot detect the assigns violation.
 *
 * Expected after fix: VERIFICATION FAILED
 *   An ASSERT(j == snapshot_j) in the inductive step should catch that j
 *   was modified even though it is not in the assigns clause.
 *
 * Current (buggy) behavior: VERIFICATION SUCCESSFUL
 *   The frame rule only generates ASSUME(j == snapshot_j), trusting the
 *   user's declaration without verifying it. The assigns violation is silently
 *   ignored.
 */

#include <assert.h>

int main()
{
  int i = 0;
  int j = 0;

  __ESBMC_loop_invariant(i >= 0 && i <= 10);
  __ESBMC_loop_assigns(i); /* Wrong: loop also modifies j! */
  while (i < 10)
  {
    i++;
    j++; /* assigns violation: j is not in the assigns clause */
  }

  return 0;
}
