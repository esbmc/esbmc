#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* ldv onenand_sim: memset of a 16 MB vmalloc buffer used to build one
 * constant per element (minutes and gigabytes in symex). */
#define SIZE (16u << 20)

int main()
{
  unsigned char *p = malloc(SIZE);
  unsigned *q = malloc(SIZE / sizeof(unsigned) * sizeof(unsigned));
  if (!p || !q)
    return 0;
  memset(p, 255, SIZE);
  memset(q, 1, SIZE);
  unsigned i = __VERIFIER_nondet_uint();
  __ESBMC_assume(i < SIZE / sizeof(unsigned));
  assert(p[i] == 255);
  assert(q[i] == 0x01010101u);
  free(p);
  free(q);
  return 0;
}
