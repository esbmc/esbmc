#include <stdlib.h>
extern int nondet_int(void);

struct S
{
  int x;
};

int main(void)
{
  struct S *s = malloc(sizeof *s);
  __ESBMC_assume(s != 0);
  s->x = 0;
  while (nondet_int())
    s->x = 5;
  __ESBMC_assert(s->x == 0 || s->x == 5, "x is 0 or 5");
  return 0;
}
