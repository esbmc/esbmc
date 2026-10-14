#include <limits.h>
#include <stdlib.h>

int nondet_int(void);

int main()
{
  int x = nondet_int();
  __ESBMC_assume(x <= -2147483647);
  return abs(x);
}
