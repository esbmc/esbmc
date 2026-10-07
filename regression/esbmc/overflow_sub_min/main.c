#include <limits.h>

int nondet_int(void);

int main()
{
  int a = nondet_int();
  __ESBMC_assume(a < 0);
  /* a - INT_MIN = a + 2^31 stays in range for every negative a. */
  int c = a - INT_MIN;
  return c;
}
