#include <assert.h>

long long nondet_ll(void);
int nondet_int(void);

int main(void)
{
  long long s = nondet_ll();
  __ESBMC_assume(0 <= s && s <= 4);
  long long sum = (-9223372036854775807LL - 1) + s;
  assert(sum > 0);
  long long diff = s - 9223372036854775807LL - 1;
  assert(diff > 0);

  int i = nondet_int();
  __ESBMC_assume(0 <= i && i <= 4);
  int isum = (-2147483647 - 1) + i;
  assert(isum > 0);
  int idiff = i - 2147483647 - 1;
  assert(idiff > 0);
  return 0;
}
