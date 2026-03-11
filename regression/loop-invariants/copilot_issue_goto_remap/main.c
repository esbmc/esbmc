/* Regression: Copilot PR #3777 — copy_loop_body GOTO target remap.
 *
 * The loop body contains a backward GOTO (label L inside the loop).
 * Branch 1 copies this body; if GOTO targets are not remapped, the copied
 * "goto L" still points into the original program, so we jump out of Branch 1
 * and never reach the inductive-step ASSERT. Then a non-inductive invariant
 * can be missed (wrong VERIFICATION SUCCESSFUL).
 *
 * Here the invariant "x == 0" holds at entry but not after one iteration
 * (x becomes 1 or more). Expected: VERIFICATION FAILED (invariant inductive step).
 * If the implementation does not remap targets, Branch 1 may not run the
 * inductive assertion and we get a wrong SUCCESSFUL.
 */
#include <assert.h>

int main(void)
{
  int x = 0;

  __ESBMC_loop_invariant(x == 0);
  while (x < 3)
  {
  L:
    x++;
    if (x < 2)
      goto L;
  }

  assert(x >= 2);
  return 0;
}
