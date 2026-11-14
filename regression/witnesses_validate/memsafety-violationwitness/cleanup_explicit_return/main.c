#include <stdlib.h>
extern int nondet_int(void);
int main(void)
{
  int *p = malloc(sizeof(int));
  __ESBMC_assume(p != 0);
  if (nondet_int())
  {
    free(p);
    return 0;
  }
  return 0;
}
