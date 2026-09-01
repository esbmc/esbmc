/* Negative counterpart of houdini_branchy_int: the loop may run zero times,
 * leaving x == 5, so x >= 6 must be reported violated. */
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
