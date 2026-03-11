/* Regression: Copilot PR #3777 — do-while entry_cond (continue condition).
 *
 * For do-while, the loop head is not "IF !cond GOTO exit", so entry_cond
 * is left nil and we always assert the inductive-step invariant after the
 * copied body. That can cause false negatives: we require INV even when the
 * copied iteration was the last one (loop would exit).
 *
 * This do-while has invariant "x <= 5". It holds at every loop head
 * (x is 0..5 when we start an iteration). At exit we do one more body
 * (x becomes 6) and then exit — so at exit x can be 6; we must not require
 * INV at that exit point. If we don't ASSUME(continue_cond) before the
 * copied body, we might assert x<=5 after a path that just exited (x=6)
 * and get a false FAILED.
 *
 * Expected: VERIFICATION SUCCESSFUL.
 */
#include <assert.h>

int main(void)
{
  int x = 0;

  __ESBMC_loop_invariant(x <= 5);
  do
  {
    x++;
  } while (x < 5);

  assert(x == 5);
  return 0;
}
