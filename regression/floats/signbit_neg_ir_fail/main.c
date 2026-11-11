#include <assert.h>
#include <math.h>
double nondet_double(void);
int main(void)
{
  double d = nondet_double();
  __ESBMC_assume(d == 0.0);
  double e = -d;
  assert(signbit(e) != signbit(-d));
  return 0;
}
