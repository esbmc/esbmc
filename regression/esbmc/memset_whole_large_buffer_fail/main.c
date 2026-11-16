#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define SIZE (16u << 20)

int main()
{
  unsigned *p = malloc(SIZE / sizeof(unsigned) * sizeof(unsigned));
  if (!p)
    return 0;
  memset(p, 1, SIZE);
  unsigned i = __VERIFIER_nondet_uint();
  __ESBMC_assume(i < SIZE / sizeof(unsigned));
  assert(p[i] == 0x01010100u);
  free(p);
  return 0;
}
