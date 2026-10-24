/* Negative counterpart of houdini_branchy_int: the loop may run zero times,
 * leaving x == 5, so x >= 6 does not hold. The claim is downstream of the
 * invariant havoc, so #7480 reports it unknown rather than failed -- but it
 * must still be reported, and must not pass. */
#include <assert.h>

int main()
{
  int x = 5;

  while (nondet_bool())
    if (x < 10)
      x = x + 1;

  assert(x >= 6);
  return 0;
}
