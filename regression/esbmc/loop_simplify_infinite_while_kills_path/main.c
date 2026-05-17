/* Regression: `while(1) { }` (and `for(;;) { }`) has a constant-false
 * exit guard. The loop body has no observable side effects, so the
 * existing erase-dead-loop path used to fire and the whole loop was
 * collapsed to nothing — letting control fall through to the assert
 * below and producing a spurious VERIFICATION FAILED.
 *
 * The fix: when the exit guard simplifies to constant false, rewrite
 * the loop head to `assume(false)` so the path is killed (correct
 * model of a non-terminating execution: no post-loop state is ever
 * reached). The assert below is then unreachable and verification
 * succeeds. */
#include <assert.h>

int main()
{
  while (1) { }
  assert(0); /* unreachable */
  return 0;
}
