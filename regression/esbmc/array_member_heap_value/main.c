#include <stdlib.h>
typedef struct { unsigned a; unsigned arr[4]; unsigned z; } S;
int main(void)
{
  S *p = malloc(sizeof(S));
  if (!p) return 0;
  p->a = 0; p->z = 0;
  unsigned k = nondet_uint();
  __ESBMC_assume(k < 4);
  unsigned *q = &p->arr[k];
  *q = 9;
  __ESBMC_assert(p->arr[k] == 9 && p->a == 0 && p->z == 0, "heap value");
  return 0;
}
