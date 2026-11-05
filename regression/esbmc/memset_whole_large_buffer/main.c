#include <assert.h>
#include <stdlib.h>
#include <string.h>

/* ldv onenand_sim: memset of a 16 MB vmalloc buffer used to build one
 * constant per byte (minutes and gigabytes in symex). */
#define SIZE (16u << 20)

int main()
{
  unsigned char *p = malloc(SIZE);
  if (!p)
    return 0;
  memset(p, 255, SIZE);
  unsigned i = __VERIFIER_nondet_uint();
  __ESBMC_assume(i < SIZE);
  assert(p[i] == 255);
  free(p);
  return 0;
}
