/*
 * loop_assigns_check_pass: Loop frame rule assigns compliance — correct case.
 *
 * The loop declares __ESBMC_loop_assigns(i), and the loop body only modifies i.
 * Variable j is NOT modified by the loop.
 *
 * Expected: VERIFICATION SUCCESSFUL
 * The assigns compliance check should PASS because the loop respects its assigns clause.
 */

#include <assert.h>

int main()
{
  int i = 0;
  int j = 42;

  __ESBMC_loop_invariant(i >= 0 && i <= 10);
  __ESBMC_loop_assigns(i);
  while (i < 10)
  {
    i++;
    /* j is never touched — loop correctly respects assigns(i) */
  }

  assert(j == 42); /* j must still be 42 */
  return 0;
}
