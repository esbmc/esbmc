/* Soundness: --termination should report a safety violation when an
 * assertion inside a non-terminating loop is reachable on the first
 * iteration. Today both master and our branch return SUCCESSFUL
 * because the --termination reduction discards in-loop assertions —
 * the rewrite of `exit_guard == false` to `assume(false)` skips the
 * body unconditionally, hiding the assert(0) the first iteration
 * would have tripped. KNOWNBUG until the termination work is
 * revisited in a dedicated branch. */
#include <assert.h>
int main()
{
  while (1)
    assert(0);
  return 0;
}
