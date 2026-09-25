/* The bodies of these primitives are linked only when a program refers to
 * them (#6831), so this test is what keeps a referenced one reaching symex. */
#include <stdlib.h>

int main(void)
{
  int a, b;
  char *p = malloc(8);
  if (!p)
    return 0;
  __ESBMC_assert(!__CPROVER_same_object(&a, &b), "distinct objects differ");
  __ESBMC_assert(__CPROVER_OBJECT_SIZE(p) == 8, "malloc(8) is 8 bytes");
  __ESBMC_assert(__CPROVER_r_ok(p, 8), "the whole allocation is readable");
  free(p);
  return 0;
}
