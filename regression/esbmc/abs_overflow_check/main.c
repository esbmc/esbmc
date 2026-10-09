#include <assert.h>
#include <limits.h>
#include <stdlib.h>

int nondet_int(void);
long nondet_long(void);

int main()
{
  int x = nondet_int();
  long y = nondet_long();
  __ESBMC_assume(x > INT_MIN && y > LONG_MIN);
  assert(abs(x) >= 0);
  assert(labs(y) >= 0);
  assert(__builtin_abs(x) == abs(x));
  return 0;
}
