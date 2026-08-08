#include <stdlib.h>

typedef struct
{
  int p;
} T;

int main(void)
{
  T *t = malloc(sizeof(T));
  if (!t)
    return 0;

  int v = nondet_int();
  if (v != 0)
  {
    t->p = 1;
  }

  /* Nothing writes t->p when v == 0, so this must not be provable. */
  __ESBMC_assert(t->p == 1, "conditionally written heap object");
  free(t);
  return 0;
}
