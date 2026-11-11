#include <stdlib.h>
typedef struct { unsigned a; unsigned arr[4]; unsigned z; } S;
int main(void)
{
  S *p = malloc(sizeof(S));
  unsigned k = nondet_uint();
  __ESBMC_assume(k < 4);
  unsigned *q = &p->arr[k];
  *q = 1;               /* p may be NULL */
  return 0;
}
