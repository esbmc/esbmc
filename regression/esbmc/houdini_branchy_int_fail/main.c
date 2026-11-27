/* Negative counterpart of houdini_branchy_int: the loop may run zero times,
 * leaving x == 5, so x >= 6 does not hold. The loop's entry state violates
 * it, so it is a real failure (#11027). */
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
