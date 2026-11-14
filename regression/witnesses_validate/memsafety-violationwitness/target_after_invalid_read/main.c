#include <stdlib.h>
int main(void)
{
  int *p = malloc(sizeof(*p));
  __ESBMC_assume(p != 0);
  *p = 1;
  free(p);
  int value = *p;
  __ESBMC_assume(value == 1);
  free(p);
}
