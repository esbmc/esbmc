#include <assert.h>
#include <math.h>

double nondet_double(void);

int main(void)
{
  double d = nondet_double();
  __ESBMC_assume(d > 0);
  assert(signbit(-d) == 0);
  return 0;
}
