/* Wider than the unrolling cap, so the symbolic-length model declines and the
   C byte loop still handles it. */
#include <string.h>
#include <assert.h>

int main()
{
  char src[96];
  char dst[96];

  for (int i = 0; i < 96; i++)
    src[i] = 5;

  unsigned long n = nondet_ulong();
  __ESBMC_assume(n == 2);

  memcpy(dst, src, n);
  assert(dst[0] == 5);
  assert(dst[1] == 5);
  return 0;
}
