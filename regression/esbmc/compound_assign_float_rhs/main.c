#include <assert.h>
#include <fenv.h>

int nondet_int(void);

int main()
{
  int i = nondet_int();
  __ESBMC_assume(i >= 0 && i <= 100);
  int d = i;
  d /= 2.5;
  assert(d == i * 2 / 5);

  long long x = 1LL << 53;
  fesetround(FE_UPWARD);
  x += 1.0;
  assert(x == (1LL << 53) + 2);
}
