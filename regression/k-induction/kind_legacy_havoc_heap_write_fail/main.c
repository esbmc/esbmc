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
  __ESBMC_assert(s->x == 0, "x is 5 after one iteration");
  return 0;
}
