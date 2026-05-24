/* Soundness: a `while (1) { break; }` loop reaches its post-loop
 * code via the break, so the trailing assert(0) is reachable and
 * must fail. Both our branch and master return SUCCESSFUL because
 * --termination's constant-false guard rewrite collapses the loop
 * to assume(false) without first checking whether the body has any
 * internal exit (break / goto / return). KNOWNBUG until the
 * termination work is revisited in a dedicated branch. */
#include <assert.h>
int main()
{
  while (1)
  {
    break;
  }
  assert(0);
  return 0;
}
