/* The property is not itself inductive: for a subnormal x, 2*x underflows and
 * 2*x - 1 rounds to exactly -1.0f, so `x > 0` does not survive one iteration.
 * `x > 1` does, and implies it. Houdini finds that by guessing from the
 * program's own constants and deleting what the solver refutes. There is no
 * counter and no affine update here, so the recogniser in
 * goto_invariant_synthesis declines this loop entirely. */
#include <assert.h>

int main()
{
  float x = 2;

  while (nondet_bool())
    x = 2 * x - 1;

  assert(x > 0);
  return 0;
}
