#include <assert.h>
double nondet_double(void);
int nondet_int(void);
int main(void)
{
  double x = nondet_double();
  __ESBMC_assume(x == 3.75);
  assert((int)x == 3);
  assert((float)x == 3.75f);
  int i = nondet_int();
  __ESBMC_assume(i == -7);
  assert((double)i == -7.0);
  unsigned u = 42u;
  assert((double)u == 42.0);
  return 0;
}
