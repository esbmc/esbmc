/* esbmc/esbmc#8681: the operand of free is checked like any expression. */
#include <stdlib.h>
int nondet_int();
int main(void)
{
  int *a[2] = {0, 0};
  int k = nondet_int();
  free(a[1 / k]);
  return 0;
}
