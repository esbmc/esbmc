#include <assert.h>

long long nondet_ll(void);

int main(void)
{
  long long a = nondet_ll();
  __ESBMC_assume(-4 <= a && a <= -1);
  long long up = a + 9223372036854775807LL + 1;
  assert(up < 0);

  long long b = nondet_ll();
  __ESBMC_assume(1 <= b && b <= 4);
  long long down = b + (-9223372036854775807LL) + (-2);
  assert(down > 0);

  long long c = nondet_ll();
  __ESBMC_assume(-1 <= c && c <= 0);
  long long prod = (c * 4611686018427387904LL) * 2;
  assert(c == 0 || prod > 0);
  return 0;
}
