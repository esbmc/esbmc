/* An integer loop with a branch in the body: not straight-line, so the affine
 * recogniser declines it, and the nondet guard means BMC never terminates on
 * its own. Houdini keeps x >= 5 and x <= 10 and drops the rest. */
#include <assert.h>

int main()
{
  int x = 5;

  while (nondet_bool())
    if (x < 10)
      x = x + 1;

  assert(x >= 5);
  return 0;
}
