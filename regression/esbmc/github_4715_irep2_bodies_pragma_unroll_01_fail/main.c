/* Negative variant of github_4715_irep2_bodies_pragma_unroll_01: the array is
 * one element too small, so the third unrolled iteration writes a[2] out of
 * the 2-element array. This in-loop array-bounds violation is reachable and
 * must still produce a counterexample (VERIFICATION FAILED) under
 * --irep2-bodies, confirming the bounds checker runs over the round-tripped
 * loop body. */
#include <stdint.h>

int main()
{
  int a[2] = {0};

  #pragma unroll 3
  for (uint32_t j = 0; j < 8; j++)
    a[j] = (int)(j + 1);

  return 0;
}
