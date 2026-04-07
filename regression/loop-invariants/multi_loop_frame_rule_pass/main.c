/*
 * multi_loop_frame_rule_pass: Two sequential loops in the same function,
 * both using the loop frame rule. Each loop respects its assigns clause.
 *
 * Guards against symbol-name collision (Issue #1/#2, Copilot review):
 * the second loop's frame_enforcert creates snapshot symbols with counter
 * starting at 0, generating the same names as the first loop
 * (e.g. __ESBMC_frame_snap_loop_0). SSA renaming in symex currently
 * prevents this from causing wrong results, but a future refactor could
 * break it. This test documents the expected correct behavior.
 *
 * Expected: VERIFICATION SUCCESSFUL
 */

#include <assert.h>

int main()
{
  /* First loop: assigns only i */
  int i = 0, a = 99;
  __ESBMC_loop_invariant(i >= 0 && i <= 5);
  __ESBMC_loop_assigns(i);
  while (i < 5)
    i++;
  assert(a == 99); /* a must be unchanged */

  /* Second loop: assigns only j — different variable, same index in snapshot counter */
  int j = 0, b = 77;
  __ESBMC_loop_invariant(j >= 0 && j <= 3);
  __ESBMC_loop_assigns(j);
  while (j < 3)
    j++;
  assert(b == 77); /* b must be unchanged */

  return 0;
}
