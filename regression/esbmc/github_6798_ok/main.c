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

  t->p = 1;

  int v = nondet_int();
  if (v != 0)
  {
    t->p = 2;
  }

  /* The join must keep both versions reachable, not just the branch's. */
  __ESBMC_assert(t->p == 1 || t->p == 2, "both join arms survive");

  int w = nondet_int();
  if (w != 0)
  {
    t->p = 3;
  }
  else
  {
    t->p = 3;
  }
  __ESBMC_assert(t->p == 3, "written on every path");

  free(t);
  return 0;
}
