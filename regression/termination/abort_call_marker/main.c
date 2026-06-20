/* Abort-call marker pattern. The loop is `while(1)` and its only
 * real exit is the call to `abort()` at iteration 10; the natural
 * loop-exit marker placed by insert_markers_for_function lives past
 * `IF !1 GOTO ...`, so it is statically unreachable from the back
 * edge. Without an additional marker IS sees no marker reachable on
 * any unwinding and reports non-termination.
 *
 * insert_abort_call_markers_for_function places an ASSERT(false)
 * immediately before every direct call to an Aborts function
 * (abort/exit/__assert_fail/...) inside a loop body. With the marker
 * in place, IS reaches it along the abort path; the forward
 * condition then closes at k = 11 because the abort short-circuits
 * every further iteration. Expected verdict: VERIFICATION
 * SUCCESSFUL. */

#include <stdlib.h>

int main()
{
  int x = 0;
  while (1)
  {
    x++;
    if (x >= 10)
      abort();
  }
  return 0;
}
