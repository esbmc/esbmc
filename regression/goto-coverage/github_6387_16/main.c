#include <assert.h>
int nondet_int();
int main()
{
  int n = nondet_int();
  __ESBMC_assume(n > 0 && n < 100);
  int s = 0;
  for (int i = 0; i < n; i++)
  {
    s += i;
    assert(s >= 0);
  }
  return 0;
}
