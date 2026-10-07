#include <limits.h>

int nondet_int(void);

int main()
{
  int a = nondet_int();
  __ESBMC_assume(a == 0);
  /* 0 - INT_MIN = 2^31 does not fit in int. */
  int c = a - INT_MIN;
  return c;
}
