#include <assert.h>
#include <limits.h>

int nondet_int(void);
long nondet_long(void);

int main()
{
  assert(__builtin_abs(-3) == 3);
  assert(__builtin_labs(-3l) == 3);
  assert(__builtin_llabs(LLONG_MIN + 1) == LLONG_MAX);

  int x = nondet_int();
  __ESBMC_assume(x != INT_MIN);
  assert(__builtin_abs(x) >= 0 && __builtin_abs(-x) == __builtin_abs(x));

  long y = nondet_long();
  __ESBMC_assume(y != LONG_MIN);
  assert(__builtin_labs(y) == (y < 0 ? -y : y));
  return 0;
}
