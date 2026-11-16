#include <assert.h>
#include <math.h>

double nondet_double(void);

int main(void)
{
  double d = nondet_double();
  __ESBMC_assume(d > 0);
  assert(signbit(-d) == 1);
  assert(signbit(fabs(-d)) == 0);
  assert(signbit(-0.0) == 1);
  return 0;
}
