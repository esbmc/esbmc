/* Negative counterpart of houdini_float_doubling: the surviving candidates
 * (x >= 2, x >= 1, x > 1) do not imply x > 3, and the loop may be skipped
 * entirely leaving x == 2, so the assertion must be reported rather than
 * absorbed by a stronger guess. The loop's entry state violates it, so it is
 * a real failure (#11027). */
#include <assert.h>

int main()
{
  float x = 2;

  while (nondet_bool())
    x = 2 * x - 1;

  assert(x > 3);
  return 0;
}
